/* A global's value type rides on its resource, as a local's does: a signed global shifts and divides
 * arithmetically, a float global negates as a float, and the dereference of a signed or float global
 * array loads its element type. The twin once typed every global as uint32 -- `(g >> 1) < 0` was
 * always false and `-g` truncated a float -- a behaviour split the structural digest cannot see. */
int32_t g_i32;
int16_t g_i16a[4];
float g_f;
double g_da[4];

uint32_t global_types(uint32_t x)
{
    g_i32 = (int32_t)x - 1000000000;
    g_i16a[0] = (int16_t)(x >> 3);
    g_f = (float)(x & 1023u) - 511.5f;
    g_da[0] = (double)(x & 4095u) - 2047.25;
    uint32_t a = (uint32_t)((g_i32 >> 1) < 0);
    uint32_t b = (uint32_t)((g_i32 / 3) < 0);
    uint32_t c = (uint32_t)((*g_i16a >> 1) < 0);
    float y = -g_f;
    double z = -*g_da;
    return a + 2u * b + 4u * c + 8u * (uint32_t)(y < -1.0f) + 16u * (uint32_t)(z < -1.0);
}
