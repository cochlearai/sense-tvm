# SPDX-License-Identifier: Apache-2.0
"""Standalone C/C++ source templates for NCNN codegen."""


def generate_header(proto_block: str) -> str:
    return f"""// SPDX-License-Identifier: Apache-2.0
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include "ncnn.h"

{proto_block}
"""


def generate_storage_section(storage_block: str) -> str:
    if not storage_block:
        return ""
    return f"""
{storage_block}
"""


def generate_runtime_helpers() -> str:
    return """
static float* g_weights = NULL;

__attribute__((noinline))
#if defined(__clang__)
__attribute__((optnone))
#endif
static void reshape_copy(const float* in, float* out, long size)
{
    if (!in || !out || in == out || size <= 0)
        return;
    volatile const float* vin = in;
    volatile float* vout = out;
    if (out < in)
    {
        for (long i = 0; i < size; ++i)
            vout[i] = vin[i];
    }
    else
    {
        for (long i = size; i-- > 0;)
            vout[i] = vin[i];
    }
}

static void max_f32(const float* a, const float* b, float* out, long size, int b_scalar)
{
    if (!a || !out || size <= 0)
        return;
    if (b_scalar)
    {
        float s = b ? b[0] : 0.0f;
        for (long i = 0; i < size; ++i)
            out[i] = a[i] > s ? a[i] : s;
    }
    else
    {
        for (long i = 0; i < size; ++i)
            out[i] = a[i] > b[i] ? a[i] : b[i];
    }
}

static void min_f32(const float* a, const float* b, float* out, long size, int b_scalar)
{
    if (!a || !out || size <= 0)
        return;
    if (b_scalar)
    {
        float s = b ? b[0] : 0.0f;
        for (long i = 0; i < size; ++i)
            out[i] = a[i] < s ? a[i] : s;
    }
    else
    {
        for (long i = 0; i < size; ++i)
            out[i] = a[i] < b[i] ? a[i] : b[i];
    }
}

static void add_channel_bias_pack4(const float* a, const float* b, float* out, int n, int c, int h, int w)
{
    if (!a || !out || !b || n <= 0 || c <= 0 || h <= 0 || w <= 0)
        return;
    int c4 = c / 4;
    int hw = h * w;
    for (int ni = 0; ni < n; ++ni)
    {
        for (int ci = 0; ci < c4; ++ci)
        {
            const float* bias = b + ci * 4;
            long base = ((long)ni * c4 + ci) * hw * 4;
            for (int i = 0; i < hw; ++i)
            {
                float* outp = out + base + i * 4;
                const float* ap = a + base + i * 4;
                outp[0] = ap[0] + bias[0];
                outp[1] = ap[1] + bias[1];
                outp[2] = ap[2] + bias[2];
                outp[3] = ap[3] + bias[3];
            }
        }
    }
}

static void concat_last_axis(const float** inputs, int num_inputs, float* output, int n, int h, int w, int c_each)
{
    if (!inputs || !output || num_inputs <= 0 || c_each <= 0)
        return;
    int c_total = num_inputs * c_each;
    for (int ni = 0; ni < n; ++ni)
    {
        for (int hi = 0; hi < h; ++hi)
        {
            for (int wi = 0; wi < w; ++wi)
            {
                long base_out = (((long)ni * h + hi) * w + wi) * c_total;
                for (int in_idx = 0; in_idx < num_inputs; ++in_idx)
                {
                    const float* in = inputs[in_idx];
                    if (!in) continue;
                    long base_in = (((long)ni * h + hi) * w + wi) * c_each;
                    for (int ci = 0; ci < c_each; ++ci)
                    {
                        output[base_out + in_idx * c_each + ci] = in[base_in + ci];
                    }
                }
            }
        }
    }
}

static void pack1_to_pack4(const float* input, float* output, int n, int c, int h, int w)
{
    if (!input || !output || c <= 0)
        return;
    int c4 = c / 4;
    for (int ni = 0; ni < n; ++ni)
    {
        for (int ci = 0; ci < c; ++ci)
        {
            int c4i = ci / 4;
            int lane = ci % 4;
            for (int hi = 0; hi < h; ++hi)
            {
                for (int wi = 0; wi < w; ++wi)
                {
                    long in_idx = ((long)ni * c + ci) * h * w + (long)hi * w + wi;
                    long out_idx = ((((long)ni * c4 + c4i) * h + hi) * w + wi) * 4 + lane;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

static void pack4_to_pack1(const float* input, float* output, int n, int c, int h, int w)
{
    if (!input || !output || c <= 0)
        return;
    int c4 = c / 4;
    for (int ni = 0; ni < n; ++ni)
    {
        for (int ci = 0; ci < c; ++ci)
        {
            int c4i = ci / 4;
            int lane = ci % 4;
            for (int hi = 0; hi < h; ++hi)
            {
                for (int wi = 0; wi < w; ++wi)
                {
                    long out_idx = ((long)ni * c + ci) * h * w + (long)hi * w + wi;
                    long in_idx = ((((long)ni * c4 + c4i) * h + hi) * w + wi) * 4 + lane;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

static void mul_scalar(const float* in, float* out, long size, float s)
{
    if (!in || !out || size <= 0)
        return;
    for (long i = 0; i < size; ++i)
        out[i] = in[i] * s;
}

static void add_channel_bias(const float* a, const float* b, float* out, int n, int c, int inner)
{
    if (!a || !b || !out || n <= 0 || c <= 0 || inner <= 0)
        return;
    for (int ni = 0; ni < n; ++ni)
    {
        for (int ci = 0; ci < c; ++ci)
        {
            float bias = b[ci];
            const float* ap = a + (size_t)(ni * c + ci) * inner;
            float* op = out + (size_t)(ni * c + ci) * inner;
            for (int i = 0; i < inner; ++i)
            {
                op[i] = ap[i] + bias;
            }
        }
    }
}

static void conv3x3_pack1(const float* input,
                          const float* weight,
                          float* output,
                          int in_c,
                          int in_h,
                          int in_w,
                          int out_c,
                          int out_h,
                          int out_w,
                          int stride,
                          int pad_top,
                          int pad_left,
                          int pad_bottom,
                          int pad_right)
{
    if (!input || !weight || !output)
        return;
    for (int oc = 0; oc < out_c; ++oc)
    {
        for (int oh = 0; oh < out_h; ++oh)
        {
            for (int ow = 0; ow < out_w; ++ow)
            {
                float sum = 0.0f;
                for (int ic = 0; ic < in_c; ++ic)
                {
                    int ih0 = oh * stride - pad_top;
                    int iw0 = ow * stride - pad_left;
                    const float* in_ptr = input + (ic * in_h * in_w);
                    const float* w_ptr = weight + ((oc * in_c + ic) * 9);
                    for (int kh = 0; kh < 3; ++kh)
                    {
                        int ih = ih0 + kh;
                        if (ih < 0 || ih >= in_h) continue;
                        for (int kw = 0; kw < 3; ++kw)
                        {
                            int iw = iw0 + kw;
                            if (iw < 0 || iw >= in_w) continue;
                            float v = in_ptr[ih * in_w + iw];
                            sum += v * w_ptr[kh * 3 + kw];
                        }
                    }
                }
                output[(oc * out_h + oh) * out_w + ow] = sum;
            }
        }
    }
}

static void depthwise3x3_pack1(const float* input,
                               const float* weight,
                               float* output,
                               int c,
                               int in_h,
                               int in_w,
                               int out_h,
                               int out_w,
                               int stride,
                               int pad_top,
                               int pad_left,
                               int pad_bottom,
                               int pad_right)
{
    if (!input || !weight || !output)
        return;
    for (int ic = 0; ic < c; ++ic)
    {
        const float* in_base = input + (ic * in_h * in_w);
        const float* w_ptr = weight + (ic * 9);
        float* out_base = output + (ic * out_h * out_w);
        for (int oh = 0; oh < out_h; ++oh)
        {
            for (int ow = 0; ow < out_w; ++ow)
            {
                float sum = 0.0f;
                int ih0 = oh * stride - pad_top;
                int iw0 = ow * stride - pad_left;
                for (int kh = 0; kh < 3; ++kh)
                {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw = 0; kw < 3; ++kw)
                    {
                        int iw = iw0 + kw;
                        if (iw < 0 || iw >= in_w) continue;
                        float v = in_base[ih * in_w + iw];
                        sum += v * w_ptr[kh * 3 + kw];
                    }
                }
                out_base[oh * out_w + ow] = sum;
            }
        }
    }
}

static void conv1x1_pack1(const float* input,
                          const float* weight,
                          float* output,
                          int in_c,
                          int in_h,
                          int in_w,
                          int out_c,
                          int out_h,
                          int out_w)
{
    if (!input || !weight || !output)
        return;
    for (int oc = 0; oc < out_c; ++oc)
    {
        const float* w_ptr = weight + oc * in_c;
        float* out_base = output + oc * out_h * out_w;
        for (int oh = 0; oh < out_h; ++oh)
        {
            for (int ow = 0; ow < out_w; ++ow)
            {
                float sum = 0.0f;
                const float* in_ptr = input + (oh * in_w + ow);
                for (int ic = 0; ic < in_c; ++ic)
                {
                    sum += in_ptr[ic * in_h * in_w] * w_ptr[ic];
                }
                out_base[oh * out_w + ow] = sum;
            }
        }
    }
}
"""


def generate_debug_helpers(dump_ir_tensor_data: bool = False, trace_operator_delay: bool = False) -> str:
    if not dump_ir_tensor_data and not trace_operator_delay:
        return ""
    sections = [
        """
static FILE* open_debug_file(const char* path)
{
    static int initialized = 0;
    if (!initialized)
    {
        mkdir("debug", 0755);
        initialized = 1;
    }
    return fopen(path, "a");
}
"""
    ]
    if dump_ir_tensor_data:
        sections.append(
            """
static void format_int_with_commas(long value, char* buf, size_t buf_size)
{
    char tmp[32];
    int len;
    int out = 0;
    snprintf(tmp, sizeof(tmp), "%ld", value);
    len = (int)strlen(tmp);
    for (int i = 0; i < len && (size_t)(out + 2) < buf_size; ++i)
    {
        if (i > 0 && ((len - i) % 3) == 0)
        {
            buf[out++] = ',';
        }
        buf[out++] = tmp[i];
    }
    buf[out] = '\\0';
}

static void dump_tensor_data(const char* name, const float* data, long size, long max_elems)
{
    FILE* f = open_debug_file("debug/op_tensor.txt");
    char size_buf[32];
    long to_print;
    if (!f || !name || !data || size <= 0)
    {
        if (f) fclose(f);
        return;
    }
    to_print = size < max_elems ? size : max_elems;
    format_int_with_commas(size, size_buf, sizeof(size_buf));
    fprintf(f, "● %s\\n", name);
    fprintf(f, "  └─ Size: %s\\n", size_buf);
    fprintf(f, "  └─ Data: [ ");
    for (long i = 0; i < to_print; ++i)
    {
        fprintf(f, "%7.4f%s", data[i], i < to_print - 1 ? ", " : "");
    }
    fprintf(f, "%s ]\\n\\n", to_print < size ? ", ..." : "");
    fclose(f);
}
"""
        )
    if trace_operator_delay:
        sections.append(
            """
static double g_total_operator_delay_ms = 0.0;
static int g_operator_delay_initialized = 0;

static void finalize_operator_delay_trace(void)
{
    FILE* f = open_debug_file("debug/op_delay.txt");
    if (!f || !g_operator_delay_initialized)
    {
        if (f) fclose(f);
        return;
    }
    fprintf(f, "-------------------------------------------------------------------------------------------\\n");
    fprintf(f, "%-56s | %9.2f ms\\n", "TOTAL", g_total_operator_delay_ms);
    fclose(f);
}

static void trace_operator_delay(const char* name, clock_t start, clock_t end)
{
    FILE* f = open_debug_file("debug/op_delay.txt");
    double elapsed_ms;
    int blocks;
    const char* colon;
    const char* op_name;
    char id_buf[32];
    if (!f || !name)
    {
        if (f) fclose(f);
        return;
    }
    if (!g_operator_delay_initialized)
    {
        fclose(f);
        f = fopen("debug/op_delay.txt", "w");
        if (!f)
        {
            return;
        }
        fprintf(f, "%-6s %-48s | %9s | %s\\n", "ID", "Operation", "Time (ms)", "Visualization");
        fprintf(f, "-------------------------------------------------------------------------------------------\\n");
        fclose(f);
        g_operator_delay_initialized = 1;
        atexit(finalize_operator_delay_trace);
        f = open_debug_file("debug/op_delay.txt");
        if (!f)
        {
            return;
        }
    }
    elapsed_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    g_total_operator_delay_ms += elapsed_ms;
    blocks = (int)(elapsed_ms * 18.0);
    if (elapsed_ms >= 0.05 && blocks == 0)
    {
        blocks = 1;
    }
    colon = strchr(name, ':');
    if (colon)
    {
        size_t id_len = (size_t)(colon - name + 1);
        if (id_len >= sizeof(id_buf))
        {
            id_len = sizeof(id_buf) - 1;
        }
        memcpy(id_buf, name, id_len);
        id_buf[id_len] = '\\0';
        op_name = colon + 1;
        while (*op_name == ' ')
        {
            ++op_name;
        }
    }
    else
    {
        snprintf(id_buf, sizeof(id_buf), "%s", name);
        op_name = "";
    }
    fprintf(f, "%-6s %-48s | %9.2f | ", id_buf, op_name, elapsed_ms);
    for (int i = 0; i < blocks; ++i)
    {
        fputs("■", f);
    }
    fputc('\\n', f);
    fclose(f);
}
"""
        )
    return "\n".join(sections)


def generate_file_size_helper() -> str:
    return """
static long file_size(FILE* f)
{
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    return size;
}
"""


def generate_inference_function(
    model_name: str,
    input_shape,
    output_shape,
    num_buffers: int,
    total_storage_mb: float,
    op_count: int,
    calls: str,
    output_src: str,
    output_elems_calc: int,
) -> str:
    return f"""
/*========================================
 * Inference Function - NCNN Standalone
 *========================================*/
int {model_name}_inference(float* input, float* output, long output_elems) {{
    /* Input: {input_shape}, Output: {output_shape} */
    /* Storages: {num_buffers}, Total: {total_storage_mb:.2f} MB */
    /* Operations: {op_count} */
    float scalar_zero = 0.0f;
    float scalar_six = 6.0f;
    float* a = input;
    float* b = input;
    float* c = output ? output : input;
    float* d = input;
    float* w = g_weights;
    int shape[4] = {{1, 1, 1, 1}};
    int axes[1] = {{0}};
    int perm[4] = {{0, 1, 2, 3}};
    long in_elems = 0;
    long out_elems_use = output ? output_elems : 0;
    const float* output_src = {output_src};
    long output_elems_calc = {output_elems_calc};

{calls}

    if (output && output_src)
    {{
        long copy_elems = output_elems;
        if (output_elems_calc > 0 && output_elems_calc < copy_elems)
            copy_elems = output_elems_calc;
        memcpy(output, output_src, (size_t)copy_elems * sizeof(float));
    }}

    return 0;
}}
"""


def generate_main_function(model_name: str, output_elems_calc: int) -> str:
    return f"""
/*========================================
 * Main Function
 *========================================*/
int main(int argc, char** argv)
{{
    int ret = 0;
    float* input = NULL;
    float* output = NULL;
    unsigned char* wbuf = NULL;
    FILE* wf = NULL;
    FILE* inf = NULL;
    FILE* outf = NULL;
    long wsize = 0;
    long in_size = 0;
    size_t wread = 0;
    size_t in_read = 0;
    long output_elems = (argc > 3) ? atol(argv[3]) : {output_elems_calc};
    const char* input_path = (argc > 1) ? argv[1] : "input.bin";
    const char* output_path = (argc > 2) ? argv[2] : "output.bin";
    const char* weights_path = (argc > 4) ? argv[4] : "weights.bin";
    clock_t infer_t0 = 0;
    clock_t infer_t1 = 0;
    double infer_ms = 0.0;

    wf = fopen(weights_path, "rb");
    if (!wf) {{
        fprintf(stderr, "failed to open %s\\n", weights_path);
        ret = 1;
        goto cleanup;
    }}
    wsize = file_size(wf);
    wbuf = (unsigned char*)malloc(wsize);
    if (!wbuf) {{
        fprintf(stderr, "malloc failed\\n");
        ret = 2;
        goto cleanup;
    }}
    wread = fread(wbuf, 1, wsize, wf);
    fclose(wf);
    wf = NULL;
    printf("Loaded weights: %ld bytes (read %zu)\\n", wsize, wread);
    g_weights = (float*)wbuf;

    inf = fopen(input_path, "rb");
    if (!inf) {{
        fprintf(stderr, "failed to open %s\\n", input_path);
        ret = 3;
        goto cleanup;
    }}
    in_size = file_size(inf);
    input = (float*)malloc(in_size);
    if (!input) {{
        fprintf(stderr, "malloc failed\\n");
        ret = 4;
        goto cleanup;
    }}
    in_read = fread(input, 1, in_size, inf);
    fclose(inf);
    inf = NULL;
    printf("Loaded input: %ld bytes (read %zu)\\n", in_size, in_read);

    if (output_elems > 0) {{
        output = (float*)malloc((size_t)output_elems * sizeof(float));
        if (!output) {{
            fprintf(stderr, "malloc failed\\n");
            ret = 5;
            goto cleanup;
        }}
        memset(output, 0, (size_t)output_elems * sizeof(float));
    }}

    infer_t0 = clock();
    ret = {model_name}_inference(input, output, output_elems);
    infer_t1 = clock();
    infer_ms = ((double)(infer_t1 - infer_t0)) / CLOCKS_PER_SEC * 1000.0;
    printf("Inference completed in %.2f ms\\n", infer_ms);
    if (ret != 0) {{
        goto cleanup;
    }}

    if (output && output_elems > 0) {{
        outf = fopen(output_path, "wb");
        if (!outf) {{
            fprintf(stderr, "failed to open %s\\n", output_path);
            ret = 6;
            goto cleanup;
        }}
        fwrite(output, sizeof(float), (size_t)output_elems, outf);
        fclose(outf);
        outf = NULL;
    }}

cleanup:
    if (wf) fclose(wf);
    if (inf) fclose(inf);
    if (outf) fclose(outf);
    if (output) free(output);
    if (input) free(input);
    if (wbuf) free(wbuf);
    g_weights = NULL;
    return ret;
}}
"""
