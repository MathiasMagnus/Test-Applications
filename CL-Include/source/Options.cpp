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
        TCLAP::ValueArg<std::size_t> platform_arg("p", "platform", "Index of platform to use", false, 0, "positive integral", cli );
        TCLAP::ValueArg<std::size_t> device_arg("d", "device", "Number of input points", false, 0, "positive integral", cli);

        std::vector<std::string> valid_dev_strings{ "all", "cpu", "gpu", "acc", "host" };
        TCLAP::ValuesConstraint<std::string> valid_dev_constraint{ valid_dev_strings };

        TCLAP::ValueArg<std::string> type_arg{ "t", "type","Type of device to use", false, "all", &valid_dev_constraint , cli };

        auto device_type = [](std::string in) -> cl_device_type
        {
            if (in == "all") return CL_DEVICE_TYPE_ALL;
            else if (in == "cpu") return CL_DEVICE_TYPE_CPU;
            else if (in == "gpu") return CL_DEVICE_TYPE_GPU;
            else if (in == "acc") return CL_DEVICE_TYPE_ACCELERATOR;
            else throw std::logic_error{ "Unkown device type after cli parse. Should not have happened." };
        };

        TCLAP::SwitchArg quiet_arg("q", "quiet", "Suppress standard output", false);
        cli.add(quiet_arg);

        cli.parse(argc, argv);

        return { length_arg.getValue(), platform_arg.getValue(), device_arg.getValue(),
                device_type(type_arg.getValue()),
                quiet_arg.getValue() };
    }
    catch (TCLAP::ArgException e)
    {
        std::stringstream ss;
        ss << e.what() << std::endl;
        ss << "error: " << e.error() << " for arg " << e.argId();
        throw cli::error{ ss.str().c_str() };
    }
    catch (std::logic_error e)
    {
        std::stringstream ss;
        ss << e.what();
        throw cli::error{ e.what() };
    }
}