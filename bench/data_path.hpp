// SPDX-License-Identifier: Apache-2.0
// Test/benchmark data root: $SHIELD_DATA_DIR (default "./data"). See docs/DATA.md for the expected layout.
#pragma once
#include <cstdlib>
#include <string>
inline std::string shvbench_data(const std::string& rel) {
    const char* e = std::getenv("SHIELD_DATA_DIR");
    return std::string(e && *e ? e : "data") + rel;
}
