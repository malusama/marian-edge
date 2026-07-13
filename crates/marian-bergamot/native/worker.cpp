#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "translator/parser.h"
#include "translator/response.h"
#include "translator/response_options.h"
#include "translator/service.h"

namespace {

constexpr std::uint32_t kMaxItems = 256;
constexpr std::uint32_t kMaxFrameBytes = 1024 * 1024;

struct Translation {
  std::string text;
  std::uint32_t input_tokens;
  std::uint32_t output_tokens;
};

bool read_u32(std::istream& input, std::uint32_t& value) {
  unsigned char bytes[4];
  if (!input.read(reinterpret_cast<char*>(bytes), sizeof(bytes))) {
    return false;
  }
  value = static_cast<std::uint32_t>(bytes[0]) |
          (static_cast<std::uint32_t>(bytes[1]) << 8) |
          (static_cast<std::uint32_t>(bytes[2]) << 16) |
          (static_cast<std::uint32_t>(bytes[3]) << 24);
  return true;
}

void write_u32(std::ostream& output, std::uint32_t value) {
  const unsigned char bytes[4] = {
      static_cast<unsigned char>(value & 0xff),
      static_cast<unsigned char>((value >> 8) & 0xff),
      static_cast<unsigned char>((value >> 16) & 0xff),
      static_cast<unsigned char>((value >> 24) & 0xff),
  };
  output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

std::string read_string(std::istream& input) {
  std::uint32_t size = 0;
  if (!read_u32(input, size) || size > kMaxFrameBytes) {
    throw std::runtime_error("invalid input frame length");
  }
  std::string text(size, '\0');
  if (!input.read(text.data(), static_cast<std::streamsize>(size))) {
    throw std::runtime_error("truncated input frame");
  }
  return text;
}

void write_string(std::ostream& output, const std::string& text) {
  if (text.size() > kMaxFrameBytes) {
    throw std::runtime_error("output frame exceeds 1 MiB");
  }
  write_u32(output, static_cast<std::uint32_t>(text.size()));
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void write_error(const std::string& message) {
  write_u32(std::cout, 1);
  write_string(std::cout, message);
  std::cout.flush();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: marian-mlx-bergamot-worker CONFIG CPU_THREADS\n";
    return 2;
  }

  try {
    const auto workers = std::stoul(argv[2]);
    if (workers == 0 || workers > 64) {
      throw std::runtime_error("CPU_THREADS must be between 1 and 64");
    }
    marian::bergamot::AsyncService::Config service_config;
    service_config.numWorkers = workers;
    service_config.logger.level = "off";
    marian::bergamot::AsyncService service(service_config);
    auto options = marian::bergamot::parseOptionsFromFilePath(argv[1]);
    auto model = service.createCompatibleModel(options);

    while (true) {
      std::uint32_t count = 0;
      if (!read_u32(std::cin, count)) {
        break;
      }
      try {
        if (count > kMaxItems) {
          throw std::runtime_error("batch exceeds 256 items");
        }
        std::vector<std::string> inputs;
        inputs.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
          inputs.push_back(read_string(std::cin));
        }

        std::vector<std::future<marian::bergamot::Response>> futures;
        futures.reserve(count);
        for (auto& input : inputs) {
          auto promise =
              std::make_shared<std::promise<marian::bergamot::Response>>();
          futures.push_back(promise->get_future());
          service.translate(
              model, std::move(input),
              [promise](marian::bergamot::Response&& response) {
                promise->set_value(std::move(response));
              },
              marian::bergamot::ResponseOptions{});
        }

        std::vector<Translation> translations;
        translations.reserve(count);
        for (auto& future : futures) {
          auto response = future.get();
          std::size_t input_tokens = 0;
          std::size_t output_tokens = 0;
          for (std::size_t sentence = 0; sentence < response.size(); ++sentence) {
            input_tokens += response.source.numWords(sentence);
            output_tokens += response.target.numWords(sentence);
          }
          if (output_tokens >= response.size()) {
            output_tokens -= response.size();
          } else {
            output_tokens = 0;
          }
          translations.push_back({
              response.getTranslatedText(),
              static_cast<std::uint32_t>(input_tokens),
              static_cast<std::uint32_t>(output_tokens),
          });
        }
        write_u32(std::cout, 0);
        write_u32(std::cout, count);
        for (const auto& translation : translations) {
          write_u32(std::cout, translation.input_tokens);
          write_u32(std::cout, translation.output_tokens);
          write_string(std::cout, translation.text);
        }
        std::cout.flush();
      } catch (const std::exception& error) {
        write_error(error.what());
      }
    }
    service.clear();
  } catch (const std::exception& error) {
    std::cerr << "Bergamot worker initialization failed: " << error.what()
              << '\n';
    return 1;
  }
  return 0;
}
