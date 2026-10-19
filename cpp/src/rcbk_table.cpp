#include "rcbk_table.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>

namespace dis {

namespace {

std::string lstrip(const std::string &s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    return start == std::string::npos ? std::string() : s.substr(start);
}

std::string strip(const std::string &s) {
    std::string out = lstrip(s);
    const auto end = out.find_last_not_of(" \t\r\n");
    return end == std::string::npos ? std::string() : out.substr(0, end + 1);
}

// Same patterns as _HEADER_PATTERNS in BK_dipole.py.
std::optional<double> header_value(const std::string &header, const std::string &pattern) {
    std::smatch match;
    if (std::regex_search(header, match, std::regex(pattern))) return std::stod(match[1].str());
    return std::nullopt;
}

}  // namespace

RcbkTable RcbkTable::read(const std::string &filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("cannot open BK file " + filename);

    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) lines.push_back(line);

    RcbkTable table;
    table.filename = filename;

    // Comment header: '#' lines before the first '###'.
    std::string header;
    for (const auto &line : lines) {
        const std::string s = lstrip(line);
        if (s.rfind("###", 0) == 0) break;
        if (s.rfind("#", 0) == 0) header += s + "\n";
    }
    const std::string number = R"(([0-9.eE+-]+))";
    table.Qs0_sq = header_value(header, R"(Q_s0\^2\s*=\s*)" + number);
    table.gamma = header_value(header, R"(\\?gamma\s*=\s*)" + number);
    table.ec = header_value(header, R"(coefficient of E inside Log is\s*)" + number);
    table.LambdaQCD = header_value(header, R"(\\?Lambda_QCD\s*=\s*)" + number);
    const auto header_x0 = header_value(header, R"(\bx0\s*=\s*)" + number);

    std::vector<std::size_t> markers;
    for (std::size_t i = 0; i < lines.size(); ++i)
        if (lstrip(lines[i]).rfind("###", 0) == 0) markers.push_back(i);
    if (markers.size() < 5)
        throw std::runtime_error(
            filename + ": expected at least 5 '###' markers (MinR, RMultiplier, RPoints, X0, "
                       "and one rapidity), found " + std::to_string(markers.size()));

    auto marker_value = [&](std::size_t i) {
        return std::stod(strip(lstrip(lines[markers[i]]).substr(3)));
    };
    table.MinR = marker_value(0);
    table.RMultiplier = marker_value(1);
    const auto RPoints = static_cast<std::size_t>(marker_value(2));
    table.x0 = marker_value(3);

    if (header_x0 && std::abs(*header_x0 - table.x0) > 1e-6 * std::abs(table.x0))
        std::cerr << "Warning: " << filename << ": x0 in the comment header (" << *header_x0
                  << ") disagrees with the '###' grid value (" << table.x0 << "); using "
                  << table.x0 << "\n";

    // Geometric r grid, exactly as the solver (and BK_dipole.py) constructs it:
    // MinR * RMultiplier**arange(RPoints) in numpy is pow(), not repeated products.
    table.r_grid.resize(RPoints);
    for (std::size_t i = 0; i < RPoints; ++i)
        table.r_grid[i] = table.MinR * std::pow(table.RMultiplier, static_cast<double>(i));

    std::vector<double> Y_values;
    std::vector<std::vector<double>> rows;
    for (std::size_t k = 4; k < markers.size(); ++k) {
        Y_values.push_back(marker_value(k));
        std::vector<double> values;
        std::size_t idx = markers[k] + 1;
        while (values.size() < RPoints && idx < lines.size()) {
            const std::string tok = strip(lines[idx++]);
            if (tok.empty() || tok[0] == '#') continue;
            values.push_back(std::stod(tok));
        }
        if (values.size() != RPoints)
            throw std::runtime_error(filename + ": rapidity block Y=" +
                                     std::to_string(Y_values.back()) + " has " +
                                     std::to_string(values.size()) + " r points, expected " +
                                     std::to_string(RPoints));
        rows.push_back(std::move(values));
    }

    // Sort the blocks by rapidity, as BK_dipole.py does.
    std::vector<std::size_t> order(Y_values.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return Y_values[a] < Y_values[b]; });
    for (std::size_t k : order) {
        table.Y_grid.push_back(Y_values[k]);
        table.N_grid.insert(table.N_grid.end(), rows[k].begin(), rows[k].end());
    }
    return table;
}

}  // namespace dis
