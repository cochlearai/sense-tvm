// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstddef>
#include <vector>

#include "cochl/include/target/nchw/ncnn.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif

struct Mat
{
    int w = 0;
    int h = 0;
    int c = 0;
    int elempack = 1;
    size_t cstep = 0;
    float* data = nullptr;

    Mat() = default;

    Mat(int _w, int _h, int _c, int _elempack, size_t _cstep, float* _data)
        : w(_w), h(_h), c(_c), elempack(_elempack), cstep(_cstep), data(_data)
    {}

    void create(int _w, int _h, int _c)
    {
        w = _w;
        h = _h;
        c = _c;
        elempack = 1;
        cstep = (size_t)w * h;
        // data pointer is managed by caller
    }

    Mat channel(int p) const
    {
        return Mat(w, h, 1, elempack, cstep, data + (size_t)p * cstep * elempack);
    }

    float* row(int y) const { return data + (size_t)y * w * elempack; }

    float operator[](int i) const { return data[i]; }

    void fill(float32x4_t v) const
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            for (int y = 0; y < h; ++y)
            {
                float* rowptr = data + (size_t)y * w * elempack;
                for (int x = 0; x < w; ++x)
                {
                    vst1q_f32(rowptr + x * elempack, v);
                }
            }
        }
        else
        {
            const float s = vgetq_lane_f32(v, 0);
            for (int y = 0; y < h; ++y)
            {
                float* rowptr = data + (size_t)y * w * elempack;
                for (int x = 0; x < w * elempack; ++x)
                {
                    rowptr[x] = s;
                }
            }
        }
#else
        (void)v;
#endif
    }

    void fill(float v) const
    {
#if __ARM_NEON
        fill(vdupq_n_f32(v));
#else
        for (int y = 0; y < h; ++y)
        {
            float* rowptr = data + (size_t)y * w * elempack;
            for (int x = 0; x < w * elempack; ++x)
            {
                rowptr[x] = v;
            }
        }
#endif
    }

    operator float*() const { return data; }
    operator const float*() const { return data; }
};

struct Option
{
    int num_threads = 1;
};

static inline float activation_ss(float v, int /*activation_type*/, const Mat& /*activation_params*/)
{
    return v;
}

#if __ARM_NEON
static inline float32x4_t activation_ps(float32x4_t v, int /*activation_type*/, const Mat& /*activation_params*/)
{
    return v;
}
#endif

// ---- copied from ncnn convolution_packed.h, adapted to local Mat/Option ----

static void convolution_transform_kernel_packed(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb

    // kernel_tm is expected to be preallocated with the correct shape

    int q = 0;
#if __ARM_NEON
#if __aarch64__
    for (; q + 7 < outch; q += 8)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;
        const float* kptr2 = (const float*)kernel + (q + 2) * inch * maxk;
        const float* kptr3 = (const float*)kernel + (q + 3) * inch * maxk;
        const float* kptr4 = (const float*)kernel + (q + 4) * inch * maxk;
        const float* kptr5 = (const float*)kernel + (q + 5) * inch * maxk;
        const float* kptr6 = (const float*)kernel + (q + 6) * inch * maxk;
        const float* kptr7 = (const float*)kernel + (q + 7) * inch * maxk;

        float* g00 = kernel_tm.channel(q / 8);

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;
                const float* k4 = kptr4 + p * maxk;
                const float* k5 = kptr5 + p * maxk;
                const float* k6 = kptr6 + p * maxk;
                const float* k7 = kptr7 + p * maxk;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    g00[4] = k4[k];
                    g00[5] = k5[k];
                    g00[6] = k6[k];
                    g00[7] = k7[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
                }
            }
        }
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;
                const float* k4 = kptr4 + p * maxk;
                const float* k5 = kptr5 + p * maxk;
                const float* k6 = kptr6 + p * maxk;
                const float* k7 = kptr7 + p * maxk;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    g00[4] = k4[k];
                    g00[5] = k5[k];
                    g00[6] = k6[k];
                    g00[7] = k7[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
                }
            }
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;
                const float* k4 = kptr4 + p * maxk;
                const float* k5 = kptr5 + p * maxk;
                const float* k6 = kptr6 + p * maxk;
                const float* k7 = kptr7 + p * maxk;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    g00[4] = k4[k];
                    g00[5] = k5[k];
                    g00[6] = k6[k];
                    g00[7] = k7[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
                }
            }
        }
        for (; p < inch; p++)
        {
            const float* k0 = kptr0 + p * maxk;
            const float* k1 = kptr1 + p * maxk;
            const float* k2 = kptr2 + p * maxk;
            const float* k3 = kptr3 + p * maxk;
            const float* k4 = kptr4 + p * maxk;
            const float* k5 = kptr5 + p * maxk;
            const float* k6 = kptr6 + p * maxk;
            const float* k7 = kptr7 + p * maxk;

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k0[k];
                g00[1] = k1[k];
                g00[2] = k2[k];
                g00[3] = k3[k];
                g00[4] = k4[k];
                g00[5] = k5[k];
                g00[6] = k6[k];
                g00[7] = k7[k];
                g00 += 8;
            }
        }
    }
#endif // __aarch64__
    for (; q + 3 < outch; q += 4)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;
        const float* kptr2 = (const float*)kernel + (q + 2) * inch * maxk;
        const float* kptr3 = (const float*)kernel + (q + 3) * inch * maxk;

#if __aarch64__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        float* g00 = kernel_tm.channel(q / 4);
#endif

        int p = 0;
#if __aarch64__
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
            }
        }
#endif // __aarch64__
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
            }
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
            }
        }
        for (; p < inch; p++)
        {
            const float* k0 = kptr0 + p * maxk;
            const float* k1 = kptr1 + p * maxk;
            const float* k2 = kptr2 + p * maxk;
            const float* k3 = kptr3 + p * maxk;

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k0[k];
                g00[1] = k1[k];
                g00[2] = k2[k];
                g00[3] = k3[k];
                g00 += 4;
            }
        }
    }
#endif // __ARM_NEON
    for (; q + 1 < outch; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;

#if __ARM_NEON
#if __aarch64__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#else
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);
#endif
#else
        float* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __ARM_NEON
#if __aarch64__
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    k0 += maxk;
                    k1 += maxk;
                    g00 += 2;
                }
            }
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    k0 += maxk;
                    k1 += maxk;
                    g00 += 2;
                }
            }
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    k0 += maxk;
                    k1 += maxk;
                    g00 += 2;
                }
            }
        }
        for (; p < inch; p++)
        {
            const float* k0 = kptr0 + p * maxk;
            const float* k1 = kptr1 + p * maxk;

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k0[k];
                g00[1] = k1[k];
                g00 += 2;
            }
        }
    }
    for (; q < outch; q++)
    {
        const float* kptr = (const float*)kernel + q * inch * maxk;

#if __ARM_NEON
#if __aarch64__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#else
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#endif
#else
        float* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __ARM_NEON
#if __aarch64__
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + p * maxk;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[k];
                    k0 += maxk;
                    g00 += 1;
                }
            }
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + p * maxk;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[k];
                    k0 += maxk;
                    g00 += 1;
                }
            }
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + p * maxk;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[k];
                    k0 += maxk;
                    g00 += 1;
                }
            }
        }
        for (; p < inch; p++)
        {
            const float* k0 = kptr + p * maxk;

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k0[k];
                g00++;
            }
        }
    }
}

// convolution_packed body is included from a separate file to keep this unit manageable
static void convolution_packed(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

    const size_t N = bottom_blob.cstep * elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;

    const size_t M = top_blob.cstep * out_elempack;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2 * elempack;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    const float* bias_data_ptr = bias_data;

    int nn_outch = 0;
    int remain_outch_start = 0;
#if __ARM_NEON
#if __aarch64__
    nn_outch = (outch - remain_outch_start) / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 8;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.channel(p / out_elempack);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float32x4_t _sum0 = vdupq_n_f32(0.f);
                float32x4_t _sum1 = vdupq_n_f32(0.f);
                float32x4_t _sum2 = vdupq_n_f32(0.f);
                float32x4_t _sum3 = vdupq_n_f32(0.f);
                float32x4_t _sum4 = vdupq_n_f32(0.f);
                float32x4_t _sum5 = vdupq_n_f32(0.f);
                float32x4_t _sum6 = vdupq_n_f32(0.f);
                float32x4_t _sum7 = vdupq_n_f32(0.f);

                if (bias_data_ptr)
                {
                    _sum0 = vld1q_f32(bias_data_ptr + p);
                    _sum1 = vld1q_f32(bias_data_ptr + p + 4);
                }

                const float* kptr = weight_data_tm.channel(p / 8);

                int q = 0;
                for (; q + 7 < inch; q += 8)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const int sok = space_ofs[k];
                        float32x4_t _r0;
                        float32x4_t _r1;
                        if (elempack == 4)
                        {
                            _r0 = vld1q_f32(r0 + sok);
                            _r1 = vld1q_f32(r0 + sok + N);
                        }
                        else
                        {
                            _r0 = float32x4_t();
                            _r1 = float32x4_t();
                            _r0 = vsetq_lane_f32(r0[sok], _r0, 0);
                            _r0 = vsetq_lane_f32(r0[sok + N], _r0, 1);
                            _r0 = vsetq_lane_f32(r0[sok + N * 2], _r0, 2);
                            _r0 = vsetq_lane_f32(r0[sok + N * 3], _r0, 3);
                            _r1 = vsetq_lane_f32(r0[sok + N * 4], _r1, 0);
                            _r1 = vsetq_lane_f32(r0[sok + N * 5], _r1, 1);
                            _r1 = vsetq_lane_f32(r0[sok + N * 6], _r1, 2);
                            _r1 = vsetq_lane_f32(r0[sok + N * 7], _r1, 3);
                        }

                        float32x4_t _w0 = vld1q_f32(kptr);
                        float32x4_t _w1 = vld1q_f32(kptr + 4);
                        float32x4_t _w2 = vld1q_f32(kptr + 4 * 2);
                        float32x4_t _w3 = vld1q_f32(kptr + 4 * 3);
                        float32x4_t _w4 = vld1q_f32(kptr + 4 * 4);
                        float32x4_t _w5 = vld1q_f32(kptr + 4 * 5);
                        float32x4_t _w6 = vld1q_f32(kptr + 4 * 6);
                        float32x4_t _w7 = vld1q_f32(kptr + 4 * 7);
                        float32x4_t _w8 = vld1q_f32(kptr + 4 * 8);
                        float32x4_t _w9 = vld1q_f32(kptr + 4 * 9);
                        float32x4_t _wa = vld1q_f32(kptr + 4 * 10);
                        float32x4_t _wb = vld1q_f32(kptr + 4 * 11);
                        float32x4_t _wc = vld1q_f32(kptr + 4 * 12);
                        float32x4_t _wd = vld1q_f32(kptr + 4 * 13);
                        float32x4_t _we = vld1q_f32(kptr + 4 * 14);
                        float32x4_t _wf = vld1q_f32(kptr + 4 * 15);
                        _sum0 = vfmaq_laneq_f32(_sum0, _w0, _r0, 0);
                        _sum1 = vfmaq_laneq_f32(_sum1, _w1, _r0, 0);
                        _sum2 = vfmaq_laneq_f32(_sum2, _w2, _r0, 1);
                        _sum3 = vfmaq_laneq_f32(_sum3, _w3, _r0, 1);
                        _sum4 = vfmaq_laneq_f32(_sum4, _w4, _r0, 2);
                        _sum5 = vfmaq_laneq_f32(_sum5, _w5, _r0, 2);
                        _sum6 = vfmaq_laneq_f32(_sum6, _w6, _r0, 3);
                        _sum7 = vfmaq_laneq_f32(_sum7, _w7, _r0, 3);
                        _sum0 = vfmaq_laneq_f32(_sum0, _w8, _r1, 0);
                        _sum1 = vfmaq_laneq_f32(_sum1, _w9, _r1, 0);
                        _sum2 = vfmaq_laneq_f32(_sum2, _wa, _r1, 1);
                        _sum3 = vfmaq_laneq_f32(_sum3, _wb, _r1, 1);
                        _sum4 = vfmaq_laneq_f32(_sum4, _wc, _r1, 2);
                        _sum5 = vfmaq_laneq_f32(_sum5, _wd, _r1, 2);
                        _sum6 = vfmaq_laneq_f32(_sum6, _we, _r1, 3);
                        _sum7 = vfmaq_laneq_f32(_sum7, _wf, _r1, 3);

                        kptr += 64;
                    }
                }
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const int sok = space_ofs[k];
                        float32x4_t _r0;
                        if (elempack == 4)
                        {
                            _r0 = vld1q_f32(r0 + sok);
                        }
                        else
                        {
                            _r0 = float32x4_t();
                            _r0 = vsetq_lane_f32(r0[sok], _r0, 0);
                            _r0 = vsetq_lane_f32(r0[sok + N], _r0, 1);
                            _r0 = vsetq_lane_f32(r0[sok + N * 2], _r0, 2);
                            _r0 = vsetq_lane_f32(r0[sok + N * 3], _r0, 3);
                        }

                        float32x4_t _w0 = vld1q_f32(kptr);
                        float32x4_t _w1 = vld1q_f32(kptr + 4);
                        float32x4_t _w2 = vld1q_f32(kptr + 4 * 2);
                        float32x4_t _w3 = vld1q_f32(kptr + 4 * 3);
                        float32x4_t _w4 = vld1q_f32(kptr + 4 * 4);
                        float32x4_t _w5 = vld1q_f32(kptr + 4 * 5);
                        float32x4_t _w6 = vld1q_f32(kptr + 4 * 6);
                        float32x4_t _w7 = vld1q_f32(kptr + 4 * 7);
                        _sum0 = vfmaq_laneq_f32(_sum0, _w0, _r0, 0);
                        _sum1 = vfmaq_laneq_f32(_sum1, _w1, _r0, 0);
                        _sum2 = vfmaq_laneq_f32(_sum2, _w2, _r0, 1);
                        _sum3 = vfmaq_laneq_f32(_sum3, _w3, _r0, 1);
                        _sum4 = vfmaq_laneq_f32(_sum4, _w4, _r0, 2);
                        _sum5 = vfmaq_laneq_f32(_sum5, _w5, _r0, 2);
                        _sum6 = vfmaq_laneq_f32(_sum6, _w6, _r0, 3);
                        _sum7 = vfmaq_laneq_f32(_sum7, _w7, _r0, 3);

                        kptr += 32;
                    }
                }
                for (; q + 1 < inch; q += 2)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        const int sok = space_ofs[k];
                        float val0;
                        float val1;
                        val0 = r0[sok];
                        val1 = r0[sok + N];

                        float32x4_t _w0 = vld1q_f32(kptr);
                        float32x4_t _w1 = vld1q_f32(kptr + 4);
                        float32x4_t _w2 = vld1q_f32(kptr + 8);
                        float32x4_t _w3 = vld1q_f32(kptr + 12);
                        _sum0 = vfmaq_n_f32(_sum0, _w0, val0);
                        _sum1 = vfmaq_n_f32(_sum1, _w1, val0);
                        _sum2 = vfmaq_n_f32(_sum2, _w2, val1);
                        _sum3 = vfmaq_n_f32(_sum3, _w3, val1);

                        kptr += 16;
                    }
                }
                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float32x4_t _val;
                        _val = vdupq_n_f32(r0[space_ofs[k]]);

                        float32x4_t _w0 = vld1q_f32(kptr);
                        float32x4_t _w1 = vld1q_f32(kptr + 4);
                        _sum0 = vfmaq_f32(_sum0, _w0, _val);
                        _sum1 = vfmaq_f32(_sum1, _w1, _val);

                        kptr += 8;
                    }
                }

                _sum0 = vaddq_f32(_sum0, _sum2);
                _sum1 = vaddq_f32(_sum1, _sum3);
                _sum4 = vaddq_f32(_sum4, _sum6);
                _sum5 = vaddq_f32(_sum5, _sum7);
                _sum0 = vaddq_f32(_sum0, _sum4);
                _sum1 = vaddq_f32(_sum1, _sum5);

                _sum0 = activation_ps(_sum0, activation_type, activation_params);
                _sum1 = activation_ps(_sum1, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    vst1q_f32(outptr, _sum0);
                    vst1q_f32(outptr + M, _sum1);
                    outptr += 4;
                }
                else
                {
                    outptr[0] = vgetq_lane_f32(_sum0, 0);
                    outptr[M] = vgetq_lane_f32(_sum0, 1);
                    outptr[M * 2] = vgetq_lane_f32(_sum0, 2);
                    outptr[M * 3] = vgetq_lane_f32(_sum0, 3);
                    outptr[M * 4] = vgetq_lane_f32(_sum1, 0);
                    outptr[M * 5] = vgetq_lane_f32(_sum1, 1);
                    outptr[M * 6] = vgetq_lane_f32(_sum1, 2);
                    outptr[M * 7] = vgetq_lane_f32(_sum1, 3);
                    outptr += 1;
                }
            }
        }
    }
    remain_outch_start += nn_outch * 8;
#endif
    nn_outch = (outch - remain_outch_start) / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 4;

        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.channel(p / out_elempack);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float32x4_t _sum0 = vdupq_n_f32(0.f);
                float32x4_t _sum1 = vdupq_n_f32(0.f);
                float32x4_t _sum2 = vdupq_n_f32(0.f);
                float32x4_t _sum3 = vdupq_n_f32(0.f);

                if (bias_data_ptr)
                {
                    _sum0 = vld1q_f32(bias_data_ptr + p);
                }

                const float* kptr = weight_data_tm.channel(p / 4);

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const int sok = space_ofs[k];
                        float32x4_t _r0;
                        if (elempack == 4)
                        {
                            _r0 = vld1q_f32(r0 + sok);
                        }
                        else
                        {
                            _r0 = float32x4_t();
                            _r0 = vsetq_lane_f32(r0[sok], _r0, 0);
                            _r0 = vsetq_lane_f32(r0[sok + N], _r0, 1);
                            _r0 = vsetq_lane_f32(r0[sok + N * 2], _r0, 2);
                            _r0 = vsetq_lane_f32(r0[sok + N * 3], _r0, 3);
                        }

                        float32x4_t _w0 = vld1q_f32(kptr);
                        float32x4_t _w1 = vld1q_f32(kptr + 4);
                        float32x4_t _w2 = vld1q_f32(kptr + 8);
                        float32x4_t _w3 = vld1q_f32(kptr + 12);
                        _sum0 = vfmaq_laneq_f32(_sum0, _w0, _r0, 0);
                        _sum1 = vfmaq_laneq_f32(_sum1, _w1, _r0, 1);
                        _sum2 = vfmaq_laneq_f32(_sum2, _w2, _r0, 2);
                        _sum3 = vfmaq_laneq_f32(_sum3, _w3, _r0, 3);

                        kptr += 16;
                    }
                }
                for (; q + 1 < inch; q += 2)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        const int sok = space_ofs[k];
                        float val0 = r0[sok];
                        float val1 = r0[sok + N];

                        float32x4_t _w0 = vld1q_f32(kptr);
                        float32x4_t _w1 = vld1q_f32(kptr + 4);
                        _sum0 = vfmaq_n_f32(_sum0, _w0, val0);
                        _sum1 = vfmaq_n_f32(_sum1, _w1, val1);

                        kptr += 8;
                    }
                }
                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float32x4_t _val = vdupq_n_f32(r0[space_ofs[k]]);
                        float32x4_t _w = vld1q_f32(kptr);
                        _sum0 = vfmaq_f32(_sum0, _w, _val);
                        kptr += 4;
                    }
                }

                _sum0 = vaddq_f32(_sum0, _sum1);
                _sum2 = vaddq_f32(_sum2, _sum3);
                _sum0 = vaddq_f32(_sum0, _sum2);
                _sum0 = activation_ps(_sum0, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    vst1q_f32(outptr, _sum0);
                    outptr += 4;
                }
                else
                {
                    outptr[0] = vgetq_lane_f32(_sum0, 0);
                    outptr[M] = vgetq_lane_f32(_sum0, 1);
                    outptr[M * 2] = vgetq_lane_f32(_sum0, 2);
                    outptr[M * 3] = vgetq_lane_f32(_sum0, 3);
                    outptr += 1;
                }
            }
        }
    }
    remain_outch_start += nn_outch * 4;
#endif

    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_data_ptr)
                    sum = bias_data_ptr[p];

                const float* kptr = weight_data_tm.channel(p);

                for (int q = 0; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        sum += r0[space_ofs[k]] * kptr[k];
                    }

                    kptr += maxk;
                }

                outptr[j] = activation_ss(sum, activation_type, activation_params);
            }

            outptr += outw;
        }
    }
}

extern "C" int convolution_transform_kernel_packed_standalone(const MatMini* kernel,
                                                               MatMini* kernel_tm,
                                                               int inch,
                                                               int outch,
                                                               int kernel_w,
                                                               int kernel_h)
{
    if (!kernel || !kernel_tm || !kernel->data || !kernel_tm->data)
        return -1;

    Mat k(kernel->w, kernel->h, kernel->c, kernel->elempack, kernel->cstep, kernel->data);
    Mat kt(kernel_tm->w, kernel_tm->h, kernel_tm->c, kernel_tm->elempack, kernel_tm->cstep, kernel_tm->data);

    convolution_transform_kernel_packed(k, kt, inch, outch, kernel_w, kernel_h);

    kernel_tm->w = kt.w;
    kernel_tm->h = kt.h;
    kernel_tm->c = kt.c;
    kernel_tm->elempack = kt.elempack;
    kernel_tm->cstep = kt.cstep;

    return 0;
}

extern "C" int convolution_packed_neon_standalone(const MatMini* bottom,
                                                   MatMini* top,
                                                   const MatMini* kernel_tm,
                                                   const MatMini* bias,
                                                   int kernel_w,
                                                   int kernel_h,
                                                   int dilation_w,
                                                   int dilation_h,
                                                   int stride_w,
                                                   int stride_h,
                                                   int activation_type)
{
    if (!bottom || !top || !kernel_tm || !bottom->data || !top->data || !kernel_tm->data)
        return -1;

    Mat bottom_m(bottom->w, bottom->h, bottom->c, bottom->elempack, bottom->cstep, bottom->data);
    Mat top_m(top->w, top->h, top->c, top->elempack, top->cstep, top->data);
    Mat kernel_tm_m(kernel_tm->w, kernel_tm->h, kernel_tm->c, kernel_tm->elempack, kernel_tm->cstep, kernel_tm->data);

    Mat bias_m;
    if (bias && bias->data)
        bias_m = Mat(bias->w, bias->h, bias->c, bias->elempack, bias->cstep, bias->data);

    Mat activation_params;
    Option opt;

    convolution_packed(bottom_m, top_m, kernel_tm_m, bias_m, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
    return 0;
}
