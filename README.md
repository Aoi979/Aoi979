# Aoi Kajitsu

Mostly C/C++ and CUDA on Linux.

Parallel computing, GPU programming, and systems.

---
## Contact

If you want to reach me, try decoding this with C++:

```cpp
template <auto X>
using C = std::integral_constant<decltype(X), X>;

consteval char decode_char(int y) {
    constexpr int A_inv = 45;
    constexpr int B = 19;
    constexpr int K = 128;

    return static_cast<char>(
        (A_inv * ((y + K) - B)) % K
    );
}

template <typename... Cs>
consteval auto decode(std::tuple<Cs...>) {
    return std::array<char, sizeof...(Cs) + 1>{
        decode_char(Cs::value)...,
        '\0'
    };
}

constexpr auto email = decode(std::tuple{
    C<24>{},  C<30>{},  C<64>{},  C<77>{},
    C<40>{},  C<40>{},  C<80>{},  C<6>{},
    C<77>{},  C<40>{},  C<43>{},  C<60>{},
    C<114>{}, C<83>{},  C<118>{}, C<84>{},
    C<24>{},  C<64>{},  C<47>{},  C<57>{},
    C<98>{},  C<30>{},  C<84>{}
});

int main() {
    std::cout << email.data() << '\n';
    return 0;
}
