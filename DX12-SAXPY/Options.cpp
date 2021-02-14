#include <Options.hpp>

// TCLAP includes
#include <tclap/CmdLine.h>

// STL includes
#include <sstream>


cli::options cli::parse(int argc, char** argv, const std::string banner)
{
    try
    {
        TCLAP::CmdLine cli(banner);

        TCLAP::ValueArg<std::size_t> length_arg("l", "length", "Length of input", false, 262144, "positive integral", cli);
        TCLAP::ValueArg<std::size_t> device_arg("d", "device", "Index of device to use", false, 0, "positive integral", cli);

        TCLAP::SwitchArg quiet_arg("q", "quiet", "Suppress standard output", false); cli.add(quiet_arg);
        TCLAP::SwitchArg debug_arg("", "debug", "Enable debug layer", false); cli.add(debug_arg);

        cli.parse(argc, argv);

        return { length_arg.getValue(), device_arg.getValue(), quiet_arg.getValue(), debug_arg.getValue() };
    }
    catch (TCLAP::ArgException e)
    {
        std::stringstream ss;
        ss << e.what() << std::endl;
        ss << "error: " << e.error() << " for arg " << e.argId();
        throw cli::error{ ss.str().c_str() };
    }
}