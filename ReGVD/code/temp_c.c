static void ac3_extract_exponents_c(uint8_t *exp, int32_t *coef, int nb_coefs)

{

    int i;



    for (i = 0; i < nb_coefs; i++) {

        int e;

        int v = abs(coef[i]);

        if (v == 0)

            e = 24;

        else {

            e = 23 - av_log2(v);

            if (e >= 24) {

                e = 24;

                coef[i] = 0;

            }

            av_assert2(e >= 0);

        }

        exp[i] = e;

    }

}
