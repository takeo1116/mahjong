// Stage03 shanten / ukeire / discard analysis fast path (C++ extension).
//
// このモジュールは Stage03 の純 Python 実装
// (mahjong_agent/baseline/shanten.py, ukeire.py, discard_select.py) と
// **同一アルゴリズム** を C++ に移植したもので、teacher rollout / encoder
// hint の wall-clock を改善する。Stage02 の旧 fast path の
// analyze_discards / find_best_discard / count_acceptance 構造を参考にしつつ、
// shanten 本体は Stage03 Python の再帰 mentsu 抽出 + 貪欲不完全面子カウント
// を逐語移植して数値一致を保証する。
//
// public API (pybind11):
//   compute_shanten(counts, meld_count=0) -> int
//   analyze_discards(counts, legal_mask, meld_count=0) -> dict
//   find_best_discard(counts, legal_mask, meld_count=0) -> dict
//
// hidden info 境界: 入力は counts / legal_mask / meld_count のみ。env / wall /
// 他家手牌 / state は一切受け取らない。
#include <array>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr int kNumTileTypes = 34;

// 么九牌 (1m,9m,1p,9p,1s,9s,東南西北白發中) の tile_type index
constexpr std::array<int, 13> kTerminalsAndHonors = {
    0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33};

int kokushi_shanten(const std::array<int, kNumTileTypes>& counts) {
    int kinds = 0;
    bool has_pair = false;
    for (int t : kTerminalsAndHonors) {
        if (counts[t] > 0) {
            kinds += 1;
            if (counts[t] >= 2) has_pair = true;
        }
    }
    return 13 - kinds - (has_pair ? 1 : 0);
}

int chiitoitsu_shanten(const std::array<int, kNumTileTypes>& counts) {
    int pairs = 0;
    int kinds = 0;
    for (int c : counts) {
        if (c >= 1) kinds += 1;
        if (c >= 2) pairs += 1;
    }
    int base = 6 - pairs;
    if (kinds < 7) base += 7 - kinds;
    return base;
}

// 残り牌から不完全面子 (対子・両面/連続・嵌張) を貪欲に数える。
// Stage03 Python _count_partial の逐語移植。
int count_partial(const std::array<int, kNumTileTypes>& counts) {
    std::array<int, kNumTileTypes> c = counts;
    int partial = 0;
    for (int suit = 0; suit < 3; ++suit) {
        int base = suit * 9;
        // 対子
        for (int i = 0; i < 9; ++i) {
            int t = base + i;
            if (c[t] >= 2) {
                c[t] -= 2;
                partial += 1;
            }
        }
        // 両面 / 連続
        for (int i = 0; i < 8; ++i) {
            int t = base + i;
            if (c[t] > 0 && c[t + 1] > 0) {
                c[t] -= 1;
                c[t + 1] -= 1;
                partial += 1;
            }
        }
        // 嵌張
        for (int i = 0; i < 7; ++i) {
            int t = base + i;
            if (c[t] > 0 && c[t + 2] > 0) {
                c[t] -= 1;
                c[t + 2] -= 1;
                partial += 1;
            }
        }
    }
    // 字牌の対子
    for (int t = 27; t < kNumTileTypes; ++t) {
        if (c[t] >= 2) partial += 1;
    }
    return partial;
}

// 完成面子抽出 (Stage03 Python _remove_groups の逐語移植)。
void remove_groups(std::array<int, kNumTileTypes>& counts, int pos, int mentsu,
                   int jantai, int& best) {
    if (mentsu >= 4) {
        int shanten = 8 - 2 * 4 - jantai;
        if (shanten < best) best = shanten;
        return;
    }

    int idx = pos;
    while (idx < kNumTileTypes && counts[idx] == 0) ++idx;

    if (idx >= kNumTileTypes) {
        int partial = count_partial(counts);
        int max_partial = 4 - mentsu;
        if (partial > max_partial) partial = max_partial;
        int shanten = 8 - 2 * mentsu - partial - jantai;
        if (shanten < best) best = shanten;
        return;
    }

    // 枝刈り
    int remaining_tiles = 0;
    for (int i = idx; i < kNumTileTypes; ++i) remaining_tiles += counts[i];
    int max_more_mentsu = remaining_tiles / 3;
    int max_total_mentsu = std::min(4, mentsu + max_more_mentsu);
    int lower_bound = 8 - 2 * max_total_mentsu - (4 - max_total_mentsu) - jantai;
    if (lower_bound >= best) return;

    // 刻子
    if (counts[idx] >= 3) {
        counts[idx] -= 3;
        remove_groups(counts, idx, mentsu + 1, jantai, best);
        counts[idx] += 3;
    }

    // 順子 (数牌のみ)
    int suit = idx / 9;
    int rel = idx % 9;
    if (suit < 3 && rel <= 6) {
        int base = suit * 9;
        if (counts[base + rel + 1] > 0 && counts[base + rel + 2] > 0) {
            counts[idx] -= 1;
            counts[base + rel + 1] -= 1;
            counts[base + rel + 2] -= 1;
            remove_groups(counts, idx, mentsu + 1, jantai, best);
            counts[idx] += 1;
            counts[base + rel + 1] += 1;
            counts[base + rel + 2] += 1;
        }
    }

    // この位置で面子を取らずに次へ
    remove_groups(counts, idx + 1, mentsu, jantai, best);
}

int regular_shanten(const std::array<int, kNumTileTypes>& counts,
                    int meld_count) {
    int best = 8 - 2 * meld_count;
    std::array<int, kNumTileTypes> c = counts;

    // 雀頭なし
    remove_groups(c, 0, meld_count, 0, best);

    // 各牌種を雀頭として
    for (int t = 0; t < kNumTileTypes; ++t) {
        if (c[t] >= 2) {
            c[t] -= 2;
            remove_groups(c, 0, meld_count, 1, best);
            c[t] += 2;
        }
    }
    return best;
}

int compute_shanten_impl(const std::array<int, kNumTileTypes>& counts,
                         int meld_count) {
    if (meld_count > 0) {
        return regular_shanten(counts, meld_count);
    }
    return std::min({regular_shanten(counts, 0), chiitoitsu_shanten(counts),
                     kokushi_shanten(counts)});
}

// 受け入れ枚数 (Stage03 count_acceptance, seen_counts=None 相当)。
// remaining = 4 - counts[t] (= 手牌側のみを既見扱い)。
int count_acceptance_impl(std::array<int, kNumTileTypes>& counts, int shanten,
                          int meld_count) {
    int total = 0;
    for (int t = 0; t < kNumTileTypes; ++t) {
        if (counts[t] >= 4) continue;
        counts[t] += 1;
        int new_sh = compute_shanten_impl(counts, meld_count);
        counts[t] -= 1;
        if (new_sh < shanten) {
            total += 4 - counts[t];
        }
    }
    return total;
}

std::array<int, kNumTileTypes> to_counts(const std::vector<int>& v) {
    if (v.size() != static_cast<size_t>(kNumTileTypes)) {
        throw std::invalid_argument("counts must be length 34");
    }
    std::array<int, kNumTileTypes> c{};
    for (int i = 0; i < kNumTileTypes; ++i) c[i] = v[i];
    return c;
}

std::array<int, kNumTileTypes> to_mask(const std::vector<int>& v) {
    if (v.size() != static_cast<size_t>(kNumTileTypes)) {
        throw std::invalid_argument("legal_mask must be length 34");
    }
    std::array<int, kNumTileTypes> m{};
    for (int i = 0; i < kNumTileTypes; ++i) m[i] = v[i];
    return m;
}

int py_compute_shanten(const std::vector<int>& counts_vec, int meld_count) {
    auto counts = to_counts(counts_vec);
    return compute_shanten_impl(counts, meld_count);
}

py::dict py_analyze_discards(const std::vector<int>& counts_vec,
                             const std::vector<int>& mask_vec, int meld_count) {
    auto counts = to_counts(counts_vec);
    auto mask = to_mask(mask_vec);

    std::vector<int> shanten_after(kNumTileTypes, -1);
    std::vector<int> acceptance(kNumTileTypes, 0);
    std::vector<float> ukeire_norm(kNumTileTypes, 0.0f);
    std::vector<float> shanten_sign(kNumTileTypes, 0.0f);

    int base_shanten = compute_shanten_impl(counts, meld_count);
    std::array<int, kNumTileTypes> c = counts;
    int max_acceptance = 0;

    for (int t = 0; t < kNumTileTypes; ++t) {
        if (c[t] < 1 || mask[t] < 1) continue;
        c[t] -= 1;
        int sh_after = compute_shanten_impl(c, meld_count);
        shanten_after[t] = sh_after;
        int acc = count_acceptance_impl(c, sh_after, meld_count);
        acceptance[t] = acc;
        if (acc > max_acceptance) max_acceptance = acc;
        int delta = base_shanten - sh_after;
        if (delta > 0) {
            shanten_sign[t] = 1.0f;
        } else if (delta < 0) {
            shanten_sign[t] = -1.0f;
        }
        c[t] += 1;
    }

    if (max_acceptance > 0) {
        for (int t = 0; t < kNumTileTypes; ++t) {
            ukeire_norm[t] = static_cast<float>(acceptance[t]) /
                             static_cast<float>(max_acceptance);
        }
    }

    py::dict d;
    d["shanten_after"] = py::cast(shanten_after);
    d["acceptance"] = py::cast(acceptance);
    d["ukeire_norm"] = py::cast(ukeire_norm);
    d["shanten_sign"] = py::cast(shanten_sign);
    return d;
}

py::dict py_find_best_discard(const std::vector<int>& counts_vec,
                              const std::vector<int>& mask_vec, int meld_count) {
    auto counts = to_counts(counts_vec);
    auto mask = to_mask(mask_vec);

    int best_shanten = 999;
    int best_acceptance = -1;
    int best_tile = -1;
    std::vector<int> best_mask(kNumTileTypes, 0);

    std::array<int, kNumTileTypes> c = counts;
    std::array<int, kNumTileTypes> sh_after_arr{};
    std::array<int, kNumTileTypes> acc_arr{};
    sh_after_arr.fill(999);
    acc_arr.fill(-1);

    for (int t = 0; t < kNumTileTypes; ++t) {
        if (mask[t] < 1 || c[t] <= 0) continue;
        c[t] -= 1;
        int sh = compute_shanten_impl(c, meld_count);
        int acc = count_acceptance_impl(c, sh, meld_count);
        sh_after_arr[t] = sh;
        acc_arr[t] = acc;
        if (sh < best_shanten ||
            (sh == best_shanten && acc > best_acceptance)) {
            best_shanten = sh;
            best_acceptance = acc;
        }
        c[t] += 1;
    }

    // 合法 discard が 1 件も無い場合 (best_shanten が初期値のまま) は、
    // 全 tile が初期値 (999 / -1) で 2nd pass を誤通過するのを防ぐため
    // early return する (Stage03 Python find_best_discard と同一挙動)。
    if (best_shanten != 999) {
        for (int t = 0; t < kNumTileTypes; ++t) {
            if (sh_after_arr[t] == best_shanten &&
                acc_arr[t] == best_acceptance) {
                best_mask[t] = 1;
                if (best_tile == -1) best_tile = t;
            }
        }
    }

    py::dict d;
    d["best_shanten"] = best_shanten;
    d["best_acceptance"] = best_acceptance;
    d["best_tile"] = best_tile;
    d["best_mask"] = py::cast(best_mask);
    return d;
}

}  // namespace

PYBIND11_MODULE(_mahjong_fast, m) {
    m.doc() = "Stage03 shanten / ukeire / discard analysis fast path (C++).";
    m.def("compute_shanten", &py_compute_shanten, py::arg("counts"),
          py::arg("meld_count") = 0,
          "34-type counts からシャンテン数を返す (Stage03 Python 同一アルゴリズム)。");
    m.def("analyze_discards", &py_analyze_discards, py::arg("counts"),
          py::arg("legal_mask"), py::arg("meld_count") = 0,
          "打牌候補ごとの shanten_after / acceptance / ukeire_norm / shanten_sign。");
    m.def("find_best_discard", &py_find_best_discard, py::arg("counts"),
          py::arg("legal_mask"), py::arg("meld_count") = 0,
          "合法打牌のうち (shanten 最小, ukeire 最大) を取る集合を返す。");
}
