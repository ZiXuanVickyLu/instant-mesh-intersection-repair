#include "accd.cuh"
#include "distance.cuh"

namespace pdspai {

    CUDA_CALLABLE bool ACCD::edge_edge_ccd(const pt &p0, const pt &p1,
                                              const pt &q0, const pt &q1,
                                              const pt &p0t, const pt &p1t,
                                              const pt &q0t, const pt &q1t,
                                        real &toi,
                                        unsigned int max_iter,
                                        real xi,
                                        real s,
                                        real tau) {
        pt vertex[4] = {p0, p1, q0, q1};
        pt displacement[4] = {p0t - p0, p1t - p1, q0t - q0, q1t - q1};
        pt eval = (displacement[0] + displacement[1] + displacement[2] + displacement[3]) / static_cast<real>(4);
        pt dis_eval[4] = {displacement[0] - eval, displacement[1] - eval,
                                                     displacement[2] - eval, displacement[3] - eval};
        real lp = max(length(dis_eval[0]), length(dis_eval[1])) +
                max(length(dis_eval[2]), length(dis_eval[3]));
        if (lp <= static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }

        real dsqr = edge_edge_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
        real dFunc = dsqr - xi * xi;
        if (dFunc <= static_cast<real>(0)) {
            real dis[4]{squaredLength(p0 - q0), squaredLength(p0 - q1), squaredLength(p1 - q0), squaredLength(p1 - q1)};
            real dis_min = dis[0];
            for (int i = 1; i < 4; ++i) dis_min = min(dis_min, dis[i]);
            dsqr = dis_min;
            dFunc = dis_min - xi * xi;
        }

        real dis_current = sqrt(dsqr);
        real g = s * dFunc / (dis_current + xi);
        toi = static_cast<real>(0);

        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            real tl = (static_cast<real>(1) - s) * dFunc / (lp * (dis_current + xi));
            ++ite_current;
            for (int i = 0; i < 4; ++i) vertex[i] += tl * dis_eval[i];
            dsqr = edge_edge_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
            dFunc = dsqr - xi * xi;
            if (dFunc <= static_cast<real>(0)) {
                real dis[4]{squaredLength(vertex[0] - vertex[2]),
                            squaredLength(vertex[1] - vertex[2]),
                            squaredLength(vertex[0] - vertex[3]),
                            squaredLength(vertex[1] - vertex[3])};
                real dis_min = dis[0];
                for (int i = 1; i < 4; ++i) dis_min = min(dis_min, dis[i]);
                dsqr = dis_min;
                dFunc = dis_min - xi * xi;
            }
            dis_current = sqrt(dsqr);
            real g_current = dFunc / (dis_current + xi);
            if (toi > static_cast<real>(0) && g_current < g) break;
            toi += tl;
            if (toi >= tau) {
                toi = static_cast<real>(1);
                return false;
            }
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

    CUDA_CALLABLE bool ACCD::vertex_triangle_ccd(const pt &p0,
                                                    const pt &q0, const pt &q1, const pt &q2,
                                                    const pt &p0t,
                                                    const pt &q0t, const pt &q1t, const pt &q2t,
                                                real &toi,
                                                unsigned int max_iter,
                                                real xi,
                                                real s,
                                                real tau) {
        pt vertex[4] = {p0, q0, q1, q2};
        pt displacement[4] = {p0t - p0, q0t - q0, q1t - q1, q2t - q2};
        pt eval = (displacement[0] + displacement[1] + displacement[2] + displacement[3]) / static_cast<real>(4);
        pt dis_eval[4] = {displacement[0] - eval, displacement[1] - eval,
                                                     displacement[2] - eval, displacement[3] - eval};
        real lp = length(dis_eval[0]) +
                max(max(length(dis_eval[1]), length(dis_eval[2])), length(displacement[3]));
        if (lp <= static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }
        real dsqr = vertex_triangle_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
        real g = s * (dsqr - xi * xi) / (sqrt(dsqr) + xi);

        toi = static_cast<real>(0);
        real tl = (static_cast<real>(1) - s) * (dsqr - xi * xi) / (lp * (sqrt(dsqr) + xi));
        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            ++ite_current;
            for (int i = 0; i < 4; ++i) vertex[i] += tl * dis_eval[i];
            dsqr = vertex_triangle_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
            real g_current = (dsqr - xi * xi) / (sqrt(dsqr) + xi);
            if (toi > static_cast<real>(0) && g_current < g) break;
            toi += tl;
            if (toi >= tau) {
                toi = static_cast<real>(1);
                return false;
            }
            tl = static_cast<real>(0.9) * g_current / lp;
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

    CUDA_CALLABLE bool ACCD::vertex_edge_ccd(const pt &p0,
                                                const pt &q0, const pt &q1,
                                                const pt &p0t,
                                                const pt &q0t, const pt &q1t,
                                            real &toi,
                                            unsigned int max_iter,
                                            real xi,
                                            real s,
                                            real tau) {
        pt vertex[3] = {p0, q0, q1};
        pt displacement[3] = {p0t - p0, q0t - q0, q1t - q1};
        pt eval = (displacement[0] + displacement[1] + displacement[2]) / static_cast<real>(3);
        pt dis_eval[3] = {displacement[0] - eval, displacement[1] - eval,
                                                     displacement[2] - eval};
        real lp = length(dis_eval[0]) + max(length(dis_eval[1]), length(dis_eval[2]));
        if (lp == static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }
        real dsqr = vertex_edge_distance_square(vertex[0], vertex[1], vertex[2]);
        real g = s * (dsqr - xi * xi) / (sqrt(dsqr) + xi);
        toi = static_cast<real>(0);
        real tl = (static_cast<real>(1) - s) * (dsqr - xi * xi) / (lp * (sqrt(dsqr) + xi));
        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            ++ite_current;
            for (int i = 0; i < 3; ++i) vertex[i] += tl * dis_eval[i];
            dsqr = vertex_edge_distance_square(vertex[0], vertex[1], vertex[2]);
            real g_current = (dsqr - xi * xi) / (sqrt(dsqr) + xi);
            if (toi > static_cast<real>(0) && g_current < g) break;
            toi += tl;
            if (toi > tau) {
                toi = static_cast<real>(1);
                return false;
            }
            tl = static_cast<real>(0.9) * g_current / lp;
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

    CUDA_CALLABLE bool ACCD::triangle_triangle_ccd(const pt &p0, const pt &p1, const pt &p2,
                                                      const pt &q0, const pt &q1, const pt &q2,
                                                      const pt &p0t, const pt &p1t, const pt &p2t,
                                                      const pt &q0t, const pt &q1t, const pt &q2t,
                                                real &toi,
                                                unsigned int max_iter,
                                                real xi,
                                                real s,
                                                real tau) {
        toi = static_cast<real>(1);
        bool res = false;
        pt p[3] = {p0, p1, p2};
        pt q[3] = {q0, q1, q2};
        pt p_t[3] = {p0t, p1t, p2t};
        pt q_t[3] = {q0t, q1t, q2t};

        for (int i = 0; i < 3; ++i) {
            int plus_i = (i + 1) % 3;
            for (int j = 0; j < 3; ++j) {
                int plus_j = (j + 1) % 3;
                real local_toi = static_cast<real>(0);
                res |= edge_edge_ccd(p[i], p[plus_i], q[j], q[plus_j],
                                    p_t[i], p_t[plus_i], q_t[j], q_t[plus_j],
                                    local_toi, xi, s, tau, max_iter);
                toi = static_cast<real>(fmin(toi, local_toi));
            }
        }

        for (int i = 0; i < 3; ++i) {
            real local_toi = static_cast<real>(0);
            res |= vertex_triangle_ccd(p[i], q[0], q[1], q[2],
                                    p_t[i], q_t[0], q_t[1], q_t[2],
                                    local_toi, xi, s, tau, max_iter);
            toi = static_cast<real>(fmin(toi, local_toi));
        }
        for (int i = 0; i < 3; ++i) {
            real local_toi = static_cast<real>(0);
            res |= vertex_triangle_ccd(q[i], p[0], p[1], p[2],
                                    q_t[i], p_t[0], p_t[1], p_t[2],
                                    local_toi, xi, s, tau, max_iter);
            toi = static_cast<real>(fmin(toi, local_toi));
        }
        return res;
    }

    CUDA_CALLABLE bool ACCD::edge_edge_ccd_on_smem(const pt &p0, const pt &p1,
                                                   const pt &q0, const pt &q1,
                                                   const pt &p0t, const pt &p1t,
                                                   const pt &q0t, const pt &q1t,
                                                   real &toi,
                                                   pt *smem,
                                                   unsigned int max_iter,
                                                   real xi,
                                                   real s,
                                                   real tau) {
        // shared layout (per query):
        // [0..3]   vertex
        // [4..7]   displacement
        // [8..11]  dis_eval
        // [12..15] scratch (reals) via reinterpret_cast ( 4 * 3 = 12 registers)
        pt *vertex = smem;
        pt *displacement = smem + 4;
        pt *dis_eval = smem + 8;
        real *scratch = reinterpret_cast<real *>(smem + 12);
        // scratch use:
        // [0] lp, [1] dsqr, [2] dFunc, [3] dis_current, [4] g, [5] tl, [6] g_current, [7] dis_min, [8..11] dis[4]

        vertex[0] = p0; vertex[1] = p1; vertex[2] = q0; vertex[3] = q1;
        displacement[0] = p0t - p0;
        displacement[1] = p1t - p1;
        displacement[2] = q0t - q0;
        displacement[3] = q1t - q1;

        pt eval = (displacement[0] + displacement[1] + displacement[2] + displacement[3]) / static_cast<real>(4);
        dis_eval[0] = displacement[0] - eval;
        dis_eval[1] = displacement[1] - eval;
        dis_eval[2] = displacement[2] - eval;
        dis_eval[3] = displacement[3] - eval;

        scratch[0] = max(length(dis_eval[0]), length(dis_eval[1])) +
                     max(length(dis_eval[2]), length(dis_eval[3]));
        if (scratch[0] <= static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }

        scratch[1] = edge_edge_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
        scratch[2] = scratch[1] - xi * xi;
        if (scratch[2] <= static_cast<real>(0)) {
            scratch[8] = squaredLength(p0 - q0);
            scratch[9] = squaredLength(p0 - q1);
            scratch[10] = squaredLength(p1 - q0);
            scratch[11] = squaredLength(p1 - q1);
            scratch[7] = scratch[8];
            for (int i = 9; i <= 11; ++i) scratch[7] = min(scratch[7], scratch[i]);
            scratch[1] = scratch[7];
            scratch[2] = scratch[7] - xi * xi;
        }

        scratch[3] = sqrt(scratch[1]);
        scratch[4] = s * scratch[2] / (scratch[3] + xi);
        toi = static_cast<real>(0);

        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            scratch[5] = (static_cast<real>(1) - s) * scratch[2] / (scratch[0] * (scratch[3] + xi));
            ++ite_current;
            for (int i = 0; i < 4; ++i) vertex[i] += scratch[5] * dis_eval[i];
            scratch[1] = edge_edge_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
            scratch[2] = scratch[1] - xi * xi;
            if (scratch[2] <= static_cast<real>(0)) {
                scratch[8] = squaredLength(vertex[0] - vertex[2]);
                scratch[9] = squaredLength(vertex[1] - vertex[2]);
                scratch[10] = squaredLength(vertex[0] - vertex[3]);
                scratch[11] = squaredLength(vertex[1] - vertex[3]);
                scratch[7] = scratch[8];
                for (int i = 9; i <= 11; ++i) scratch[7] = min(scratch[7], scratch[i]);
                scratch[1] = scratch[7];
                scratch[2] = scratch[7] - xi * xi;
            }
            scratch[3] = sqrt(scratch[1]);
            scratch[6] = scratch[2] / (scratch[3] + xi);
            if (toi > static_cast<real>(0) && scratch[6] < scratch[4]) break;
            toi += scratch[5];
            if (toi >= tau) {
                toi = static_cast<real>(1);
                return false;
            }
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

    CUDA_CALLABLE bool ACCD::vertex_triangle_ccd_on_smem(const pt &p0,
                                                         const pt &q0, const pt &q1, const pt &q2,
                                                         const pt &p0t,
                                                         const pt &q0t, const pt &q1t, const pt &q2t,
                                                         real &toi,
                                                         pt *smem,
                                                         unsigned int max_iter,
                                                         real xi,
                                                         real s,
                                                         real tau) {
        // shared layout (per query):
        // [0..3]   vertex
        // [4..7]   displacement
        // [8..11]  dis_eval
        // [12..15] scratch (reals)
        pt *vertex = smem;
        pt *displacement = smem + 4;
        pt *dis_eval = smem + 8;
        real *scratch = reinterpret_cast<real *>(smem + 12);
        // scratch: [0] lp, [1] dsqr, [2] g, [3] tl, [4] g_current

        vertex[0] = p0; vertex[1] = q0; vertex[2] = q1; vertex[3] = q2;
        displacement[0] = p0t - p0;
        displacement[1] = q0t - q0;
        displacement[2] = q1t - q1;
        displacement[3] = q2t - q2;

        pt eval = (displacement[0] + displacement[1] + displacement[2] + displacement[3]) / static_cast<real>(4);
        dis_eval[0] = displacement[0] - eval;
        dis_eval[1] = displacement[1] - eval;
        dis_eval[2] = displacement[2] - eval;
        dis_eval[3] = displacement[3] - eval;

        scratch[0] = length(dis_eval[0]) +
                     max(max(length(dis_eval[1]), length(dis_eval[2])), length(displacement[3]));
        if (scratch[0] <= static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }
        scratch[1] = vertex_triangle_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
        scratch[2] = s * (scratch[1] - xi * xi) / (sqrt(scratch[1]) + xi);

        toi = static_cast<real>(0);
        scratch[3] = (static_cast<real>(1) - s) * (scratch[1] - xi * xi) /
                     (scratch[0] * (sqrt(scratch[1]) + xi));
        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            ++ite_current;
            for (int i = 0; i < 4; ++i) vertex[i] += scratch[3] * dis_eval[i];
            scratch[1] = vertex_triangle_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
            scratch[4] = (scratch[1] - xi * xi) / (sqrt(scratch[1]) + xi);
            if (toi > static_cast<real>(0) && scratch[4] < scratch[2]) break;
            toi += scratch[3];
            if (toi >= tau) {
                toi = static_cast<real>(1);
                return false;
            }
            scratch[3] = static_cast<real>(0.9) * scratch[4] / scratch[0];
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

    CUDA_CALLABLE bool ACCD::vertex_edge_ccd_on_smem(const pt &p0,
                                                     const pt &q0, const pt &q1,
                                                     const pt &p0t,
                                                     const pt &q0t, const pt &q1t,
                                                     real &toi,
                                                     pt *smem,
                                                     unsigned int max_iter,
                                                     real xi,
                                                     real s,
                                                     real tau) {
        // shared layout (per query):
        // [0..3]   vertex (0..2 used)
        // [4..7]   displacement (4..6 used)
        // [8..11]  dis_eval (8..10 used)
        // [12..15] scratch (reals)
        pt *vertex = smem;
        pt *displacement = smem + 4;
        pt *dis_eval = smem + 8;
        real *scratch = reinterpret_cast<real *>(smem + 12);
        // scratch: [0] lp, [1] dsqr, [2] g, [3] tl, [4] g_current

        vertex[0] = p0; vertex[1] = q0; vertex[2] = q1;
        displacement[0] = p0t - p0;
        displacement[1] = q0t - q0;
        displacement[2] = q1t - q1;

        pt eval = (displacement[0] + displacement[1] + displacement[2]) / static_cast<real>(3);
        dis_eval[0] = displacement[0] - eval;
        dis_eval[1] = displacement[1] - eval;
        dis_eval[2] = displacement[2] - eval;

        scratch[0] = length(dis_eval[0]) + max(length(dis_eval[1]), length(dis_eval[2]));
        if (scratch[0] == static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }
        scratch[1] = vertex_edge_distance_square(vertex[0], vertex[1], vertex[2]);
        scratch[2] = s * (scratch[1] - xi * xi) / (sqrt(scratch[1]) + xi);
        toi = static_cast<real>(0);
        scratch[3] = (static_cast<real>(1) - s) * (scratch[1] - xi * xi) / (scratch[0] * (sqrt(scratch[1]) + xi));
        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            ++ite_current;
            for (int i = 0; i < 3; ++i) vertex[i] += scratch[3] * dis_eval[i];
            scratch[1] = vertex_edge_distance_square(vertex[0], vertex[1], vertex[2]);
            scratch[4] = (scratch[1] - xi * xi) / (sqrt(scratch[1]) + xi);
            if (toi > static_cast<real>(0) && scratch[4] < scratch[2]) break;
            toi += scratch[3];
            if (toi > tau) {
                toi = static_cast<real>(1);
                return false;
            }
            scratch[3] = static_cast<real>(0.9) * scratch[4] / scratch[0];
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

    CUDA_CALLABLE bool ACCD::triangle_triangle_ccd_on_smem(const pt &p0, const pt &p1, const pt &p2,
                                                           const pt &q0, const pt &q1, const pt &q2,
                                                           const pt &p0t, const pt &p1t, const pt &p2t,
                                                           const pt &q0t, const pt &q1t, const pt &q2t,
                                                           real &toi,
                                                           pt *smem,
                                                           unsigned int max_iter,
                                                           real xi,
                                                           real s,
                                                           real tau) {
        toi = static_cast<real>(1);
        bool res = false;
        pt p[3] = {p0, p1, p2};
        pt q[3] = {q0, q1, q2};
        pt p_t[3] = {p0t, p1t, p2t};
        pt q_t[3] = {q0t, q1t, q2t};

        for (int i = 0; i < 3; ++i) {
            int plus_i = (i + 1) % 3;
            for (int j = 0; j < 3; ++j) {
                int plus_j = (j + 1) % 3;
                real local_toi = static_cast<real>(0);
                res |= edge_edge_ccd_on_smem(p[i], p[plus_i], q[j], q[plus_j],
                                             p_t[i], p_t[plus_i], q_t[j], q_t[plus_j],
                                             local_toi, smem, max_iter, xi, s, tau);
                toi = static_cast<real>(fmin(toi, local_toi));
            }
        }

        for (int i = 0; i < 3; ++i) {
            real local_toi = static_cast<real>(0);
            res |= vertex_triangle_ccd_on_smem(p[i], q[0], q[1], q[2],
                                               p_t[i], q_t[0], q_t[1], q_t[2],
                                               local_toi, smem, max_iter, xi, s, tau);
            toi = static_cast<real>(fmin(toi, local_toi));
        }
        for (int i = 0; i < 3; ++i) {
            real local_toi = static_cast<real>(0);
            res |= vertex_triangle_ccd_on_smem(q[i], p[0], p[1], p[2],
                                               q_t[i], p_t[0], p_t[1], p_t[2],
                                               local_toi, smem, max_iter, xi, s, tau);
            toi = static_cast<real>(fmin(toi, local_toi));
        }
        return res;
    }

} // namespace pdspai