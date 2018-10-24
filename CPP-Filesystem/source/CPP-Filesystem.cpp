#include <iostream>
#include <filesystem>
#include <algorithm>
#include <numeric>

namespace fs = std::filesystem;

int main()
{
    fs::path dir = fs::canonical(".");
    
    // Seems about right, but gives wrong result. Directory iterators are InputIterators,
    // hence they implement shallow-copy semantics (sotre a shared_ptr), meaning creating
    // a copy will result in shared state. Search algorithms typically return a copy of an
    // intermediate state and carry on. When using shallow-copy objects, the result is
    // "destroyed" by advancing the iterator further. (Result will always be last, not end)
    //
    // This shallow copy behavior is useful when advancing the iterator doesn't invalidate
    // all previous copies holding previous states. (Previous values of InputIterator are
    // otherwise meaningless.) This behavior is not useful in this case.
    auto latest_max_elem = std::max_element(fs::recursive_directory_iterator{ dir },
                                            fs::recursive_directory_iterator{},
                                            [](const fs::directory_entry& lhs,
                                               const fs::directory_entry& rhs)
    {
        return fs::last_write_time(lhs.path()) < fs::last_write_time(rhs.path());
    });

    // If ultimately we'll be interested in the name of the last written filesystem entry
    // we could persist std::filesystem::path, instead of the iterator itself.
    auto latest_reduce = std::reduce(fs::recursive_directory_iterator{ dir },
                                     fs::recursive_directory_iterator{},
                                     dir,
                                     [](const fs::path& path,
                                        const fs::directory_entry& entry)
    {
        return entry.last_write_time() > fs::last_write_time(path) ? entry.path() : path;
    });
    
    if (latest_max_elem != fs::recursive_directory_iterator{})
        std::cout << "latest file (std::max_element): " << latest_max_elem->path() << std::endl;
    else
        std::cerr << "something weird happened" << std::endl;

    std::cout << "latest file (std::reduce): " << latest_reduce << std::endl;

    return 0;
}
