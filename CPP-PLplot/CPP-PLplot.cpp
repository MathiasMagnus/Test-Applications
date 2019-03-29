// STL includes
#include <vector>
#include <cmath>
#include <thread>
#include <iostream>

// PLplot includes
#include <plplot/plstream.h>

int main(int argc, char** argv)
{
    int x_min = 0;
    int x_max = 100;
    PLFLT a = 100;
    PLFLT b = 25;
    PLFLT c = (x_max - x_min) / 10;
    PLFLT d = 10;

    std::vector<PLFLT> x(x_max - x_min), y(x_max - x_min);

    // Give values to plot point x coordinates
    for (auto i = x_min; i < x.size(); ++i) x.at(i) = i;

    // Create chart with 1 plot in x and 1 plot in y dimensions and Window renderer
    plstream plot_2d(1, 1, "wingcc");

    // Parse potential PLplot CLI options
    plot_2d.parseopts(&argc, argv, PL_PARSE_FULL);

    // Initialize PLplot internals
    plot_2d.init();

    // Set sub-plot to use
    plot_2d.adv(0);

    // Set aspect ratio for the plot
    plot_2d.vasp(std::pow(16.0 / 9.0, -1));

    // Set window world coordinates
    plot_2d.wind(0, x_max, 0, a + d);

    //plot_2d.env(x_min, x_max, d, a + d, 0, 0);

    for (auto I = 1.0; I > 0.01; I -= 0.001)
    {
        // Sleep to avoid flicker and simulate expensive iteration
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Update data
        for (auto i = x_min; i < x.size(); ++i) y.at(i) = I * a * std::exp(-std::pow(i - b, 2) / (2 * std::pow(c, 2))) + d;

        // Clear window
        plot_2d.clear();

        // Draw plot title and axis labels
        plot_2d.lab("x", "y", "Generic Gaussian");

        // Draw axis
        plot_2d.box("abcfnt", 0, 0, "abcfnt", 0, 0);

        // Draw data
        plot_2d.line(static_cast<PLINT>(x.size()), x.data(), y.data());

        // Render
        plot_2d.flush();
    }

    return 0;
}