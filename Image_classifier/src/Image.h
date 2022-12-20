#include <vector>

typedef unsigned char uint_8;


class Image {
    public:
        Image(std::vector<uint_8>& data);

    private:
        std::vector<uint_8> red_channel;
        std::vector<uint_8> green_channel;
        std::vector<uint_8> blue_channel;
};