// Helpers shared by several unit test files: reading the reference tables
// written by reference/generate_reference.py and comparing columns of them.

#ifndef TESTS_TEST_HELPERS_H_
#define TESTS_TEST_HELPERS_H_

#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"

// A whitespace-separated table. Lines starting with '#' are comments; the
// last comment line before the data names the columns.
class ReferenceTable {
public:
    explicit ReferenceTable(const std::string &name) {
        const std::string path = std::string(DIS_REFERENCE_DIR) + "/" + name;
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open reference file " + path);
        std::string header;
        for (std::string line; std::getline(in, line);) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                header = line.substr(1);
                continue;
            }
            std::istringstream fields(line);
            std::vector<std::string> row;
            for (std::string f; fields >> f;) row.push_back(f);
            rows_.push_back(row);
        }
        std::istringstream names(header);
        for (std::string n; names >> n;) columns_[n] = columns_.size();
        for (const auto &row : rows_)
            if (row.size() != columns_.size())
                throw std::runtime_error(path + ": row length does not match the header");
    }

    std::size_t size() const { return rows_.size(); }

    double operator()(std::size_t row, const std::string &column) const {
        return std::stod(text(row, column));
    }

    const std::string &text(std::size_t row, const std::string &column) const {
        const auto it = columns_.find(column);
        if (it == columns_.end()) throw std::runtime_error("no column " + column);
        return rows_.at(row)[it->second];
    }

    std::vector<double> column(const std::string &name) const {
        std::vector<double> out(size());
        for (std::size_t i = 0; i < size(); ++i) out[i] = (*this)(i, name);
        return out;
    }

private:
    std::map<std::string, std::size_t> columns_;
    std::vector<std::vector<std::string>> rows_;
};

// Checks |got[i] - want[i]| <= atol + rtol |want[i]| for every row, reporting
// the worst row on failure instead of one failure per row.
inline void check_columns_close(const std::string &what, const std::vector<double> &got,
                                const std::vector<double> &want, double rtol, double atol) {
    REQUIRE(got.size() == want.size());
    double worst = 0;
    std::size_t worst_row = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double excess = std::abs(got[i] - want[i]) / (atol + rtol * std::abs(want[i]));
        if (!(excess <= worst)) {  // also catches NaN
            worst = std::isnan(excess) ? INFINITY : excess;
            worst_row = i;
        }
    }
    INFO(what << ": worst row " << worst_row << " got " << got[worst_row] << " want "
              << want[worst_row] << " (rtol " << rtol << ", atol " << atol << ")");
    CHECK(worst <= 1.0);
}

inline double max_abs(const std::vector<double> &v) {
    double m = 0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

#endif  // TESTS_TEST_HELPERS_H_
