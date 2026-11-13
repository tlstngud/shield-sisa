// Build-time helper: embed a file as `extern "C" const unsigned char <name>[]; const size_t <name>_size;`
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: bin2c <in> <out.cpp> <symbol>\n");
        return 2;
    }
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;
    std::vector<unsigned char> d;
    int c;
    while ((c = fgetc(f)) != EOF) d.push_back((unsigned char)c);
    fclose(f);
    FILE* o = fopen(argv[2], "wb");
    if (!o) return 1;
    fprintf(o, "#include <cstddef>\nextern \"C\" const unsigned char %s[] = {\n", argv[3]);
    for (size_t i = 0; i < d.size(); i++) fprintf(o, "%u,%s", d[i], (i % 32 == 31) ? "\n" : "");
    fprintf(o, "0};\nextern \"C\" const size_t %s_size = %zu;\n", argv[3], d.size());
    fclose(o);
    return 0;
}
