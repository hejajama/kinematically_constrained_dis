// Physical constants, as in src/small_x_physics/building_blocks/constants.py.

#ifndef DIS_CONSTANTS_HPP
#define DIS_CONSTANTS_HPP

namespace dis {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double alpha_em = 1.0 / 137.035999084;
constexpr int Nc = 3;
constexpr double CF = (Nc * Nc - 1.0) / (2.0 * Nc);
constexpr double LambdaQCD = 0.241;  // GeV

}  // namespace dis

#endif  // DIS_CONSTANTS_HPP
