#pragma once

// STL includes
#include <cstddef>      // std::size_t
#include <string>       // std::string


namespace cli
{
    struct options
    {
        std::size_t length, dev_id;
        bool quiet;
    };

    options parse(int argc, char** argv, const std::string banner);

    class error
    {
    public:

        error() = default;
        error(const error&) = default;
        error(error&&) = default;
        ~error() = default;

        error(std::string message) : m_message(message) {}

        const char* what() { return m_message.c_str(); }

    private:

        std::string m_message;
    };
}