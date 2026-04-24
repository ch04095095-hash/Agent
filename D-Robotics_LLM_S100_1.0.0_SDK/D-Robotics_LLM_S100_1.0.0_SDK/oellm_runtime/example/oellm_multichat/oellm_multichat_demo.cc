// Copyright (c) [2025] [Horizon Robotics][Horizon Bole].
//
// You can use this software according to the terms and conditions of
// the Apache v2.0.
// You may obtain a copy of Apache v2.0. at:
//
//     http: //www.apache.org/licenses/LICENSE-2.0
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See Apache v2.0 for more details.

#include <arpa/inet.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "xlm.h"  // NOLINT

using json = nlohmann::json;

const std::unordered_map<int32_t, xlm_infer_backend> kModelTypeMap = {
    {0, XLM_INFER_BACKEND_BPU_0},
    {1, XLM_INFER_BACKEND_BPU_1},
    {2, XLM_INFER_BACKEND_BPU_2},
    {3, XLM_INFER_BACKEND_BPU_3},
};

static std::atomic<bool> g_running{true};

static void on_signal(int) { g_running.store(false); }

static char *read_chat_template_file(const char *filepath) {
  FILE *file = fopen(filepath, "r");
  if (!file) {
    std::cout << "open " << filepath << " failed" << std::endl;
    return nullptr;
  }

  if (fseek(file, 0, SEEK_END) != 0) {
    std::cout << "seek end failed for " << filepath << std::endl;
    fclose(file);
    return nullptr;
  }
  long tell_pos = ftell(file);
  if (tell_pos < 0) {
    std::cout << "ftell failed for " << filepath << std::endl;
    fclose(file);
    return nullptr;
  }
  if (static_cast<unsigned long>(tell_pos) >
      std::numeric_limits<size_t>::max() - 1) {
    std::cout << "file too large: " << filepath << std::endl;
    fclose(file);
    return nullptr;
  }
  size_t filesize = static_cast<size_t>(tell_pos);
  if (fseek(file, 0, SEEK_SET) != 0) {
    std::cout << "seek set failed for " << filepath << std::endl;
    fclose(file);
    return nullptr;
  }
  char *content = reinterpret_cast<char *>(malloc(filesize + 1));
  if (!content) {
    std::cout << "malloc mem for content failed" << std::endl;
    fclose(file);
    return nullptr;
  }

  size_t read_size = fread(content, 1, filesize, file);
  if (read_size != filesize) {
    std::cout << "read file failed: " << filepath << std::endl;
    free(content);
    fclose(file);
    return nullptr;
  }
  content[filesize] = '\0';

  fclose(file);
  return content;
}

static const char *kDecisionSystemPrompt = R"PROMPT(你是矿车控制决策模块。只输出一个合法 JSON。

输入为 current、window、event_state、history；window.signals 里的每个信号只包含 value 和 meaning，meaning 是主要语义依据。
决策原则：硬底线触发 -> BRAKE；ready_to_move -> MOVE；异常/过高/过低/接近报警 -> HOLD。
输出字段仅允许 action, risk_level, reason, suspected_fault, recommended_adjustment, monitor_next, confidence。
reason 要像工程师判断：先说总体状态，再说关键风险，再说动作建议。
只输出 JSON，不要多余文本。

最小合法模板:
{"action":"HOLD","risk_level":"warning","reason":"","suspected_fault":[],"recommended_adjustment":[],"monitor_next":[],"confidence":0.5}
)PROMPT";

static std::string extract_json_candidate(const std::string &text) {
  const auto first = text.find('{');
  const auto last = text.rfind('}');
  if (first == std::string::npos || last == std::string::npos || last <= first) {
    return "";
  }
  return text.substr(first, last - first + 1);
}

static bool is_array_of_strings(const json &arr) {
  if (!arr.is_array()) return false;
  for (const auto &item : arr) {
    if (!item.is_string()) return false;
  }
  return true;
}

static bool validate_event_state(const json &event_state, std::string &reason) {
  if (!event_state.is_object()) {
    reason = "event_state must be object";
    return false;
  }
  for (const auto &field : {"primary_event", "severity", "reason"}) {
    if (!event_state.contains(field) || !event_state[field].is_string()) {
      reason = std::string("event_state missing/invalid field: ") + field;
      return false;
    }
  }
  if (event_state.contains("secondary_events") &&
      !is_array_of_strings(event_state["secondary_events"])) {
    reason = "event_state.secondary_events must be string array";
    return false;
  }
  if (event_state.contains("recommended_adjustments") &&
      !is_array_of_strings(event_state["recommended_adjustments"])) {
    reason = "event_state.recommended_adjustments must be string array";
    return false;
  }
  if (event_state.contains("confidence_hint") &&
      !event_state["confidence_hint"].is_number()) {
    reason = "event_state.confidence_hint must be number";
    return false;
  }
  if (event_state.contains("semantic_summary") &&
      !is_array_of_strings(event_state["semantic_summary"])) {
    reason = "event_state.semantic_summary must be string array";
    return false;
  }
  return true;
}

static bool is_valid_decision_json(const json &obj, std::string &reason) {
  static const std::vector<std::string> required_fields = {
      "action", "risk_level", "reason", "suspected_fault",
      "recommended_adjustment", "monitor_next", "confidence"};
  for (const auto &field : required_fields) {
    if (!obj.contains(field)) {
      reason = "missing field: " + field;
      return false;
    }
  }

  if (!obj["action"].is_string() || !obj["risk_level"].is_string() ||
      !obj["reason"].is_string()) {
    reason = "string field invalid";
    return false;
  }
  const auto action = obj["action"].get<std::string>();
  const auto risk_level = obj["risk_level"].get<std::string>();
  if (action != "MOVE" && action != "HOLD" && action != "BRAKE" &&
      action != "FAULT_REPORT") {
    reason = "invalid action";
    return false;
  }
  if (risk_level != "normal" && risk_level != "warning" &&
      risk_level != "danger") {
    reason = "invalid risk_level";
    return false;
  }
  for (const auto &field : {"suspected_fault", "recommended_adjustment",
                             "monitor_next"}) {
    if (!obj[field].is_array() || !is_array_of_strings(obj[field])) {
      reason = std::string(field) + " must be string array";
      return false;
    }
  }
  if (!obj["confidence"].is_number()) {
    reason = "confidence must be number";
    return false;
  }
  const double confidence = obj["confidence"].get<double>();
  if (confidence < 0.0 || confidence > 1.0) {
    reason = "confidence out of range";
    return false;
  }
  if (!obj.empty() && obj.size() != 7) {
    reason = "unexpected extra fields";
    return false;
  }
  return true;
}

static json default_decision_json() {
  return json{{"action", "HOLD"},
              {"risk_level", "warning"},
              {"reason", "模型输出不合法，已兜底"},
              {"suspected_fault", json::array()},
              {"recommended_adjustment", json::array()},
              {"monitor_next", json::array()},
              {"confidence", 0.5}};
}

static bool parse_and_validate_decision(const std::string &raw, json *parsed,
                                       std::string *reason) {
  const std::string candidate = extract_json_candidate(raw);
  if (candidate.empty()) {
    if (reason) *reason = "no json candidate";
    return false;
  }
  try {
    json obj = json::parse(candidate);
    if (!obj.is_object()) {
      if (reason) *reason = "json is not object";
      return false;
    }
    std::string validate_reason;
    if (!is_valid_decision_json(obj, validate_reason)) {
      if (reason) *reason = validate_reason;
      return false;
    }
    if (parsed) *parsed = obj;
    return true;
  } catch (const std::exception &e) {
    if (reason) *reason = e.what();
    return false;
  }
}

struct DecisionRunContext {
  std::mutex mu;
  std::condition_variable cv;
  std::string raw_output;
  bool finished{false};
  bool timed_out{false};
  bool started{false};
  bool error{false};
  std::string error_text;
};

static void callback(xlm_result_s *result, xlm_state_e state, void *userdata) {
  auto *ctx = reinterpret_cast<DecisionRunContext *>(userdata);
  if (!ctx) return;
  std::lock_guard<std::mutex> lk(ctx->mu);
  ctx->started = true;
  if (state == XLM_STATE_ERROR) {
    ctx->error = true;
    ctx->error_text = "xlm runtime error";
    ctx->finished = true;
    ctx->cv.notify_all();
    return;
  }
  if (result && result->text) {
    ctx->raw_output += result->text;
  }
  if (state == XLM_STATE_END) {
    ctx->finished = true;
    ctx->cv.notify_all();
  }
}

static std::string build_decision_prompt(const json &user_input) {
  const json current = user_input.value("current", json::object());
  const json window = user_input.value("window", json::object());
  const json event_state = user_input.value("event_state", json::object());
  const json history = user_input.value("history", json::object());

  json compact;
  compact["current"] = current;
  compact["window"] = window;
  compact["event_state"] = event_state;
  compact["history"] = history;

  std::string prompt;
  prompt += "你是矿车控制决策模块。只输出一行合法 JSON，对象字段固定为 action, risk_level, reason, suspected_fault, recommended_adjustment, monitor_next, confidence。\n";
  prompt += "输入已经按 current/window/event_state/history 结构化。\n";
  prompt += "window.signals 里每个信号只含 value 和 meaning，meaning 是最重要的语义依据。\n";
  prompt += "优先依据 event_state，其次参考 window 中各信号的 meaning，最后参考 current/history。\n";
  prompt += "规则：ready_to_move -> MOVE；*_over_stop / emergency_stop / heartbeat_lost -> BRAKE；异常/过高/过低/接近报警 -> HOLD。\n";
  prompt += "不要输出解释、markdown、代码块、前后缀、闲聊。\n";
  prompt += "reason 要像工程师判断，先说总体状态，再说关键风险，再说动作建议。\n";
  prompt += "confidence 必须在 0 到 1 之间。\n";
  prompt += "最小模板:{\"action\":\"HOLD\",\"risk_level\":\"warning\",\"reason\":\"\",\"suspected_fault\":[],\"recommended_adjustment\":[],\"monitor_next\":[],\"confidence\":0.5}\n";
  prompt += "输入:" + compact.dump() + "\n";
  prompt += "只输出 JSON:";
  return prompt;
}

static std::string http_response(const std::string &status,
                                 const std::string &body,
                                 const std::string &content_type = "application/json") {
  std::string resp;
  resp += "HTTP/1.1 " + status + "\r\n";
  resp += "Content-Type: " + content_type + "\r\n";
  resp += "Connection: close\r\n";
  resp += "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
  resp += body;
  return resp;
}

class SimpleHttpServer {
 public:
  using Handler = std::function<std::string(const std::string &, const std::string &)>;

  SimpleHttpServer(int port, Handler handler, Handler health_handler)
      : port_(port), handler_(std::move(handler)), health_handler_(std::move(health_handler)) {}

  bool run() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      std::perror("socket");
      return false;
    }
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
      std::perror("bind");
      close(fd);
      return false;
    }
    if (listen(fd, 64) < 0) {
      std::perror("listen");
      close(fd);
      return false;
    }
    server_fd_ = fd;
    std::cout << "[Server] listening on 0.0.0.0:" << port_ << std::endl;

    while (g_running.load()) {
      sockaddr_in cli{};
      socklen_t len = sizeof(cli);
      int cfd = accept(fd, reinterpret_cast<sockaddr *>(&cli), &len);
      if (cfd < 0) {
        if (!g_running.load()) break;
        continue;
      }
      std::thread(&SimpleHttpServer::handle_client, this, cfd).detach();
    }
    close(fd);
    return true;
  }

 private:
  void handle_client(int cfd) {
    std::string req;
    char buf[4096];
    ssize_t n = 0;
    while ((n = recv(cfd, buf, sizeof(buf), 0)) > 0) {
      req.append(buf, buf + n);
      if (req.find("\r\n\r\n") != std::string::npos) break;
    }
    if (req.empty()) {
      close(cfd);
      return;
    }

    const auto line_end = req.find("\r\n");
    const std::string request_line = req.substr(0, line_end);
    const bool is_get = request_line.rfind("GET ", 0) == 0;
    const bool is_post = request_line.rfind("POST ", 0) == 0;

    std::string path = "/";
    {
      const auto p1 = request_line.find(' ');
      const auto p2 = request_line.find(' ', p1 + 1);
      if (p1 != std::string::npos && p2 != std::string::npos) {
        path = request_line.substr(p1 + 1, p2 - p1 - 1);
      }
    }

    std::string body;
    if (is_get && path == "/health") {
      body = health_handler_("GET", "");
      const auto resp = http_response("200 OK", body);
      send(cfd, resp.data(), resp.size(), 0);
      close(cfd);
      return;
    }

    if (!is_post || path != "/infer") {
      body = "{\"error\":\"not found\"}";
      const auto resp = http_response("404 Not Found", body);
      send(cfd, resp.data(), resp.size(), 0);
      close(cfd);
      return;
    }

    const std::string header_sep = "\r\n\r\n";
    const auto body_pos = req.find(header_sep);
    std::string req_body = body_pos == std::string::npos ? "" : req.substr(body_pos + header_sep.size());
    std::string resp_body = handler_("POST", req_body);
    const auto resp = http_response("200 OK", resp_body);
    send(cfd, resp.data(), resp.size(), 0);
    close(cfd);
  }

  int port_;
  int server_fd_{-1};
  Handler handler_;
  Handler health_handler_;
};

struct WorkerState {
  bool busy{false};
  bool shutdown{false};
  std::string last_error;
  int64_t last_submit_ms{0};
  int64_t last_success_ms{0};
  int64_t last_elapsed_ms{0};
  int consecutive_timeouts{0};
  std::string last_request_id;
  json last_decision;
};

class SingleQueueWorker {
 public:
  using InferFn = std::function<json(const std::string &, int)>;

  SingleQueueWorker(InferFn infer_fn, int timeout_ms)
      : infer_fn_(std::move(infer_fn)), timeout_ms_(timeout_ms) {}

  void start() {
    thread_ = std::thread([this]() { run(); });
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      state_.shutdown = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }

  bool submit(const std::string &request_id, const std::string &payload, json *out) {
    std::unique_lock<std::mutex> lk(mu_);
    if (state_.busy) {
      state_.last_error = "worker busy";
      return false;
    }
    state_.busy = true;
    state_.last_submit_ms = now_ms();
    state_.last_request_id = request_id;
    pending_payload_ = payload;
    pending_out_ = out;
    cv_.notify_one();
    done_cv_.wait(lk, [&]() { return !state_.busy || state_.shutdown; });
    if (pending_out_ && !state_.shutdown) {
      *pending_out_ = result_;
    }
    return !state_.shutdown;
  }

  WorkerState snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    WorkerState snap;
    snap.busy = state_.busy;
    snap.shutdown = state_.shutdown;
    snap.last_error = state_.last_error;
    snap.last_submit_ms = state_.last_submit_ms;
    snap.last_success_ms = state_.last_success_ms;
    snap.last_elapsed_ms = state_.last_elapsed_ms;
    snap.consecutive_timeouts = state_.consecutive_timeouts;
    snap.last_request_id = state_.last_request_id;
    snap.last_decision = state_.last_decision;
    return snap;
  }

 private:
  static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  void run() {
    while (true) {
      std::string payload;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]() { return state_.shutdown || state_.busy; });
        if (state_.shutdown) break;
        payload = pending_payload_;
      }

      auto start = std::chrono::steady_clock::now();
      json decision = infer_fn_(payload, timeout_ms_);
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start).count();

      {
        std::lock_guard<std::mutex> lk(mu_);
        result_ = decision;
        state_.last_decision = decision;
        state_.last_elapsed_ms = elapsed;
        if (decision.contains("reason") &&
            std::string(decision.value("reason", "")).find("timeout") != std::string::npos) {
          state_.consecutive_timeouts += 1;
          state_.last_error = "timeout";
        } else {
          state_.consecutive_timeouts = 0;
          state_.last_error.clear();
          state_.last_success_ms = now_ms();
        }
        state_.busy = false;
      }
      done_cv_.notify_all();
    }
  }

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  InferFn infer_fn_;
  int timeout_ms_;
  std::thread thread_;
  mutable WorkerState state_;
  std::string pending_payload_;
  json *pending_out_{nullptr};
  json result_;
};

static void print_usage() {
  std::cout << "usage:\n"
            << "  ./oellm_multichat -c <config_file.json> [--mode http] [--port 18081]\n"
            << "  ./oellm_multichat -c <config_file.json> [--mode stdin]  # deprecated, forced to http\n";
}

int32_t main(int32_t argc, char **argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  std::string config_path;
  std::string mode = "http";
  int port = 18081;

  for (int32_t i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage();
      return 0;
    }
    if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
      config_path = argv[++i];
      continue;
    }
    if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
      continue;
    }
    if (arg == "--port" && i + 1 < argc) {
      port = std::atoi(argv[++i]);
      continue;
    }
  }

  if (config_path.empty()) {
    std::cout << "Usage: " << argv[0] << " -c <config_file.json> [--mode http] [--port 18081]" << std::endl;
    return 1;
  }

  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    std::cout << "failed to open json file: " << config_path << std::endl;
    return 1;
  }

  json config;
  try {
    config_file >> config;
  } catch (const std::exception &e) {
    std::cout << "error parsing json: " << e.what() << std::endl;
    return 1;
  }

  const std::vector<std::string> required_fields = {"hbm_path", "tokenizer_dir", "model_type", "template_path"};
  for (const auto &field : required_fields) {
    if (!config.contains(field)) {
      std::cout << "missing required field: " << field << std::endl;
      return 1;
    }
  }

  if (!config["hbm_path"].is_string() || !config["tokenizer_dir"].is_string() ||
      !config["template_path"].is_string() || !config["model_type"].is_number_integer()) {
    std::cout << "invalid config field type" << std::endl;
    return 1;
  }

  std::string hbm_path = config["hbm_path"].get<std::string>();
  std::string tokenizer_dir = config["tokenizer_dir"].get<std::string>();
  std::string template_path = config["template_path"].get<std::string>();
  int32_t model_type = config["model_type"].get<int32_t>();
  int32_t bpu_core = -1;
  if (config.contains("bpu_core")) {
    if (!config["bpu_core"].is_number_integer()) {
      std::cout << "field 'bpu_core' must be integer type" << std::endl;
      return 1;
    }
    bpu_core = config["bpu_core"].get<int32_t>();
  }

  std::cout << "hbm_path: " << hbm_path << std::endl;
  std::cout << "tokenizer_dir: " << tokenizer_dir << std::endl;
  std::cout << "template_path: " << template_path << std::endl;
  std::cout << "model_type: " << model_type << std::endl;
  std::cout << "bpu_core: " << bpu_core << std::endl;
  std::cout << "mode: " << mode << std::endl;
  std::cout << "port: " << port << std::endl;

  xlm_common_params_t param = xlm_create_default_param();
  param.model_path = hbm_path.c_str();
  param.token_config_path = tokenizer_dir.c_str();
  param.model_type = static_cast<xlm_model_type>(model_type);
  param.k_cache_int8 = false;

  xlm_handle_t llm_handle = nullptr;
  int32_t ret = xlm_init(&param, callback, &llm_handle);
  if (ret != 0) {
    std::cout << "xlm init failed" << std::endl;
    return 1;
  }
  std::cout << "xlm init success" << std::endl;

  if (bpu_core == -1) {
    std::cout << "infer backend: ANY" << std::endl;
  } else {
    std::cout << "infer backend: " << bpu_core << std::endl;
  }

  char *chat_template = read_chat_template_file(template_path.c_str());
  if (!chat_template) {
    std::cout << "failed to load chat template" << std::endl;
    xlm_destroy(&llm_handle);
    return 1;
  }

  xlm_input_s input;
  memset(&input, 0, sizeof(input));
  input.request_num = 1;
  std::vector<xlm_lm_request_t> requests(input.request_num);
  input.requests = requests.data();

  auto &request = input.requests[0];
  memset(&request, 0, sizeof(request));
  request.type = XLM_INPUT_PROMPT;
  request.new_chat = true;
  request.system_prompt = kDecisionSystemPrompt;
  request.chat_template = chat_template;
  request.infer_backend = (bpu_core == -1) ? XLM_INFER_BACKEND_BPU_ANY : kModelTypeMap.at(bpu_core);

  auto infer_once = [&](const std::string &payload_text, int timeout_ms) -> json {
    json user_input;
    try {
      user_input = json::parse(payload_text);
    } catch (const std::exception &e) {
      return json{{"error", std::string("input parse error: ") + e.what()}};
    }
    if (!user_input.is_object()) {
      return json{{"error", "input must be JSON object"}};
    }
    if (user_input.contains("event_state")) {
      std::string reason;
      if (!validate_event_state(user_input["event_state"], reason)) {
        return json{{"error", std::string("invalid event_state: ") + reason}};
      }
    }

    std::string prompt = build_decision_prompt(user_input);
    request.prompt = prompt.c_str();

    DecisionRunContext ctx;
    std::thread infer_thread([&]() { xlm_infer(llm_handle, &input, &ctx); });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lk(ctx.mu);
    const bool finished = ctx.cv.wait_until(lk, deadline, [&]() { return ctx.finished; });
    if (!finished) {
      ctx.timed_out = true;
      lk.unlock();
      if (infer_thread.joinable()) infer_thread.detach();
      json fallback = default_decision_json();
      fallback["reason"] = "fallback: inference timeout";
      fallback["timeout_ms"] = timeout_ms;
      return fallback;
    }
    lk.unlock();
    if (infer_thread.joinable()) infer_thread.join();

    json parsed;
    std::string parse_reason;
    if (!parse_and_validate_decision(ctx.raw_output, &parsed, &parse_reason)) {
      parsed = default_decision_json();
      parsed["reason"] = std::string("fallback: ") + parse_reason;
    }
    return parsed;
  };

  SingleQueueWorker worker(infer_once, 5000);
  worker.start();

  auto health = [&]() -> std::string {
    const WorkerState s = worker.snapshot();
    return json{{"status", "ok"},
                {"service", "oellm_multichat"},
                {"busy", s.busy},
                {"last_error", s.last_error},
                {"last_submit_ms", s.last_submit_ms},
                {"last_success_ms", s.last_success_ms},
                {"last_elapsed_ms", s.last_elapsed_ms},
                {"consecutive_timeouts", s.consecutive_timeouts},
                {"last_request_id", s.last_request_id}}
        .dump();
  };

  if (mode != "http") {
    std::cout << "mode '" << mode << "' is deprecated; forcing http mode" << std::endl;
  }

  auto handler = [&](const std::string &method, const std::string &body) -> std::string {
    const auto start = std::chrono::steady_clock::now();
    if (body.empty()) {
      return json{{"error", "empty body"}}.dump();
    }
    json decision;
    if (!worker.submit(method + ":infer", body, &decision)) {
      decision = default_decision_json();
      decision["reason"] = "worker busy";
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    decision["elapsed_ms"] = elapsed;
    decision["request_method"] = method;
    return decision.dump();
  };

  auto health_handler = [&](const std::string &, const std::string &) -> std::string {
    return health();
  };

  SimpleHttpServer server(port, handler, health_handler);
  server.run();

  worker.stop();
  free(chat_template);
  chat_template = nullptr;
  ret = xlm_destroy(&llm_handle);
  return ret;
}
