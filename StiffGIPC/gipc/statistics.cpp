#include <gipc/statistics.h>
#include <fstream>
namespace gipc
{
Statistics::Statistics() {}

void Statistics::write_to_file(const std::string& filename)
{
    std::ofstream file;
    // Report open, write and close errors instead of silently losing evidence.
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(filename);
    file << m_json.dump(4);
    file.close();
}
}  // namespace gipc
