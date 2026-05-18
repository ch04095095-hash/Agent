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

#include <memory>
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

static const char *kDecisionSystemPrompt = R"PROMPT(只输出一个合法 JSON。)PROMPT";

static std::string sanitize_utf8_for_json(const std::string &input);
static void sanitize_json_strings(json *value);
static std::string safe_json_dump(json value);

static std::string extract_json_candidate(const std::string &text) {
  const auto first = text.find('{');
  const auto last = text.rfind('}');
  if (first == std::string::npos || last == std::string::npos || last <= first) {
    return "";
  }
  return text.substr(first, last - first + 1);
}

static std::string repair_json_candidate(std::string candidate) {
  const std::string broken_risk = "\"risk_level:\"";
  size_t broken_pos = 0;
  while ((broken_pos = candidate.find(broken_risk, broken_pos)) != std::string::npos) {
    candidate.replace(broken_pos, broken_risk.size(), "\"risk_level\":\"");
    broken_pos += std::string("\"risk_level\":\"").size();
  }

  std::vector<std::pair<std::string, std::string>> replacements = {
      {"\"riskLevel\"", "\"risk_level\""},
      {"\"overallLevel\"", "\"overall_level\""},
      {"\"suspect_fault\"", "\"suspected_fault\""},
      {"\"suspectFault\"", "\"suspected_fault\""},
      {"\"recommendedAdjustment\"", "\"recommended_adjustment\""},
      {"\"monitorNext\"", "\"monitor_next\""},
      {"\"warningTags\"", "\"warning_tags\""}};
  for (const auto &item : replacements) {
    size_t pos = 0;
    while ((pos = candidate.find(item.first, pos)) != std::string::npos) {
      candidate.replace(pos, item.first.size(), item.second);
      pos += item.second.size();
    }
  }
  return candidate;
}

static bool parse_and_validate_decision(const std::string &raw, json *parsed,
                                       std::string *reason) {
  const std::string candidate = extract_json_candidate(raw);
  if (candidate.empty()) {
    if (reason) *reason = "no json candidate";
    return false;
  }
  const std::string sanitized_candidate = sanitize_utf8_for_json(candidate);
  try {
    json obj = json::parse(repair_json_candidate(sanitized_candidate));
    if (!obj.is_object()) {
      if (reason) *reason = "json is not object";
      return false;
    }
    sanitize_json_strings(&obj);
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

static std::string xlm_state_name(xlm_state_e state) {
  switch (state) {
    case XLM_STATE_ERROR:
      return "ERROR";
    case XLM_STATE_END:
      return "END";
    default:
      return "STATE_" + std::to_string(static_cast<int>(state));
  }
}

static void callback(xlm_result_s *result, xlm_state_e state, void *userdata) {
  if (!userdata) {
    if (state == XLM_STATE_ERROR) {
      std::cout << "[LLM] callback error" << std::endl;
    } else if (state == XLM_STATE_END) {
      std::cout << std::endl << "[LLM] callback end" << std::endl;
    } else if (result && result->text) {
      std::cout << result->text << std::flush;
    }
    return;
  }

  auto *ctx = reinterpret_cast<DecisionRunContext *>(userdata);
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
  return user_input.dump();
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

static std::string sanitize_stdio_output(std::string text) {
  text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
  return text;
}

 static std::string sanitize_utf8_for_json(const std::string &input) {
  std::string out;
  out.reserve(input.size());
  for (size_t i = 0; i < input.size();) {
    const unsigned char c = static_cast<unsigned char>(input[i]);
    if (c < 0x80) {
      out.push_back(static_cast<char>(c));
      ++i;
      continue;
    }

    size_t len = 0;
    if ((c & 0xE0) == 0xC0) {
      len = 2;
    } else if ((c & 0xF0) == 0xE0) {
      len = 3;
    } else if ((c & 0xF8) == 0xF0) {
      len = 4;
    } else {
      out.push_back('?');
      ++i;
      continue;
    }

    if (i + len > input.size()) {
      out.push_back('?');
      break;
    }

    bool valid = true;
    for (size_t j = 1; j < len; ++j) {
      const unsigned char cc = static_cast<unsigned char>(input[i + j]);
      if ((cc & 0xC0) != 0x80) {
        valid = false;
        break;
      }
    }

    if (!valid) {
      out.push_back('?');
      ++i;
      continue;
    }

    out.append(input, i, len);
    i += len;
  }
  return out;
}

static void sanitize_json_strings(json *value) {
  if (value == nullptr) return;
  if (value->is_string()) {
    *value = sanitize_utf8_for_json(value->get<std::string>());
    return;
  }
  if (value->is_object()) {
    for (auto it = value->begin(); it != value->end(); ++it) {
      sanitize_json_strings(&it.value());
    }
    return;
  }
  if (value->is_array()) {
    for (auto &item : *value) {
      sanitize_json_strings(&item);
    }
  }
}

static std::string safe_json_dump(json value) {
  sanitize_json_strings(&value);
  return value.dump();
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
      const auto header_end = req.find("\r\n\r\n");
      if (header_end != std::string::npos) {
        size_t content_length = 0;
        std::string headers = req.substr(0, header_end);
        std::string lower_headers = headers;
        std::transform(lower_headers.begin(), lower_headers.end(), lower_headers.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const auto cl_pos = lower_headers.find("content-length:");
        if (cl_pos != std::string::npos) {
          const auto value_start = cl_pos + std::string("content-length:").size();
          const auto value_end = lower_headers.find("\r\n", value_start);
          const std::string value = lower_headers.substr(value_start, value_end == std::string::npos ? std::string::npos : value_end - value_start);
          content_length = static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
        }
        if (req.size() >= header_end + 4 + content_length) break;
      }
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
  param.context_size = 512;
  param.sampling.min_keep = 1;
  param.sampling.min_p = 0.0f;
  param.sampling.temp = 0.0f;
  param.sampling.top_k = 1;
  param.sampling.top_p = 1.0f;
  param.sampling.typ_p = 1.0f;
  param.sampling.penalty_last_n = 64;
  param.sampling.penalty_freq = 0.0f;
  param.sampling.penalty_present = 0.0f;
  param.sampling.penalty_repeat = 1.2f;
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

  xlm_input_t input;
  memset(&input, 0, sizeof(input));
  input.request_num = 1;
  auto requests = std::unique_ptr<xlm_lm_request_t, decltype(&free)>(
      reinterpret_cast<xlm_lm_request_t *>(malloc(sizeof(xlm_lm_request_t) * input.request_num)), free);
  if (!requests) {
    std::cout << "malloc requests failed" << std::endl;
    free(chat_template);
    xlm_destroy(&llm_handle);
    return 1;
  }
  input.requests = requests.get();

  xlm_lm_request_t *request = &input.requests[0];
  memset(request, 0, sizeof(xlm_lm_request_t));
  request->type = XLM_INPUT_PROMPT;
  request->new_chat = true;
  request->system_prompt = nullptr;
  request->chat_template = chat_template;
  request->infer_backend = (bpu_core == -1) ? XLM_INFER_BACKEND_BPU_ANY : kModelTypeMap.at(bpu_core);

  const bool enable_llm_decode = std::getenv("OELLM_AGENT_ENABLE_LLM") != nullptr;
  std::cout << "agent_llm_decode: " << (enable_llm_decode ? "enabled" : "disabled") << std::endl;

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
    if (!enable_llm_decode) {
      return json{{"error", "llm decode disabled"},
                  {"hint", "set OELLM_AGENT_ENABLE_LLM=1 to enable model generation"}};
    }

    char input_str[10240];
    memset(input_str, 0, sizeof(input_str));
    const std::string prompt = build_decision_prompt(user_input);
    std::snprintf(input_str, sizeof(input_str), "%s", prompt.c_str());
    request->new_chat = true;
    request->system_prompt = nullptr;
    request->chat_template = chat_template;
    request->prompt = input_str;

    DecisionRunContext ctx;
    std::cout << "[LLM] infer begin prompt_len=" << std::strlen(input_str) << std::endl;
    const int infer_ret = xlm_infer(llm_handle, &input, &ctx);
    std::cout << "[LLM] xlm_infer ret=" << infer_ret << std::endl;
    request->new_chat = true;
    if (infer_ret != 0) {
      return json{{"error", "xlm_infer failed"}, {"ret", infer_ret}};
    }

    std::string raw_output;
    bool has_error = false;
    std::string error_text;
    {
      std::lock_guard<std::mutex> lk(ctx.mu);
      raw_output = ctx.raw_output;
      has_error = ctx.error;
      error_text = ctx.error_text;
    }

    raw_output = sanitize_utf8_for_json(raw_output);
    if (!error_text.empty()) {
      error_text = sanitize_utf8_for_json(error_text);
    }

    if (has_error) {
      return json{{"raw_output", raw_output},
                  {"llm_prompt_mode", "template_minimal_user_prompt_sync"},
                  {"error", error_text.empty() ? "xlm runtime error" : error_text}};
    }

    json parsed;
    std::string parse_reason;
    if (!parse_and_validate_decision(raw_output, &parsed, &parse_reason)) {
      return json{{"raw_output", raw_output},
                  {"llm_prompt_mode", "template_minimal_user_prompt_sync"},
                  {"error", parse_reason}};
    }
    parsed["llm_prompt_mode"] = "template_minimal_user_prompt_sync";
    parsed["raw_output"] = raw_output;
    return parsed;
  };

  SingleQueueWorker worker(infer_once, 5000);
  worker.start();

  auto health = [&]() -> std::string {
    const WorkerState s = worker.snapshot();
    return safe_json_dump(json{{"status", "ok"},
                               {"service", "oellm_multichat"},
                               {"busy", s.busy},
                               {"last_error", s.last_error},
                               {"last_submit_ms", s.last_submit_ms},
                               {"last_success_ms", s.last_success_ms},
                               {"last_elapsed_ms", s.last_elapsed_ms},
                               {"consecutive_timeouts", s.consecutive_timeouts},
                               {"last_request_id", s.last_request_id}});
  };

  auto handler = [&](const std::string &method, const std::string &body) -> std::string {
    const auto start = std::chrono::steady_clock::now();
    if (body.empty()) {
      return json{{"error", "empty body"}}.dump();
    }
    json decision;
    if (!worker.submit(method + ":infer", body, &decision)) {
      decision = json{{"error", "worker busy"}};
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    decision["elapsed_ms"] = elapsed;
    decision["request_method"] = method;
    return safe_json_dump(std::move(decision));
  };

  auto health_handler = [&](const std::string &, const std::string &) -> std::string {
    return health();
  };

  if (mode == "direct") {
    char direct_prompt[10240];
    memset(direct_prompt, 0, sizeof(direct_prompt));
    std::snprintf(direct_prompt, sizeof(direct_prompt), "%s", "你好，请只回答一个词：OK");
    request->new_chat = true;
    request->system_prompt = nullptr;
    request->chat_template = chat_template;
    request->prompt = direct_prompt;
    std::cout << "[Direct] infer begin prompt_len=" << std::strlen(direct_prompt) << std::endl;
    const int infer_ret = xlm_infer_async(llm_handle, &input, nullptr);
    std::cout << "[Direct] xlm_infer_async ret=" << infer_ret << std::endl;
    if (infer_ret == 0) {
      std::this_thread::sleep_for(std::chrono::seconds(20));
    }
  } else if (mode == "stdin") {
    std::cout << "[User] <<<" << std::endl;
    std::string line;
    while (g_running.load() && std::getline(std::cin, line)) {
      line = sanitize_stdio_output(line);
      if (line == "exit") {
        break;
      }
      if (line.empty() || line == "reset") {
        std::cout << "[User] <<<" << std::endl;
        continue;
      }
      const std::string response = handler("STDIN", line);
      std::cout << "[Assistant] >>>" << std::endl;
      std::cout << response << std::endl;
      std::cout << "[User] <<<" << std::endl;
    }
  } else {
    SimpleHttpServer server(port, handler, health_handler);
    server.run();
  }

  worker.stop();
  free(chat_template);
  chat_template = nullptr;
  ret = xlm_destroy(&llm_handle);
  return ret;
}
