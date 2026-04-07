#include <complex.h>
int main(void) {
    double complex z = 1.0 + 2.0*I;
    z = cexp(z);
    return 0;
}
