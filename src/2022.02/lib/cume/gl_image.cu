#include "gl_image.h"

#ifdef LIBPNGPP
class pixel_generator : public png::generator< png::rgba_pixel, pixel_generator> {
public:
    pixel_generator(ImagePoint* pixels, size_t width, size_t height)
		: png::generator< png::rgba_pixel, pixel_generator >(width, height), m_pixels(pixels), m_width(width), m_height(height)
    {
    }

    png::byte* get_next_row(size_t pos)
    {
        return (png::byte*) (m_pixels + pos*m_width);
    }

private:
    ImagePoint* m_pixels;
    size_t m_width, m_height;
};
#else
class pixel_generator {
public:
    pixel_generator(ImagePoint* pixels, size_t width, size_t height)
		: m_pixels(pixels), m_width(width), m_height(height)
    {
    }

    byte* get_next_row(size_t pos)
    {
        return (byte*) (m_pixels + pos*m_width);
    }

private:
    ImagePoint* m_pixels;
    size_t m_width, m_height;
};

#endif

GLImage::GLImage(int width, int height) {
		m_width = width;
		m_height = height;
		m_size = width * height * sizeof(ImagePoint);
		m_pixels = new ImagePoint[ width * height ];
		memset(m_pixels, 0, m_size);
}

void GLImage::save(std::string file_name) {
		std::ofstream out(file_name, std::ios_base::binary);
		out<<"P6\n" // <- ASCII header, then binary RGB data
				<< m_width <<" "<< m_height << "\n"
				<<"255\n"; // maximum channel value is 255 (so 8-bit channels)
		for (int y = 0; y < m_height; ++y) {
			for (int x = 0; x < m_width; ++x) {
				int offset = y * m_width + x;
				out << m_pixels[offset].red << m_pixels[offset].green << m_pixels[offset].blue;
			}
		}
		out.close();
	}

#ifdef LIBPNGPP
void GLImage::save_png(std::string file_name) {
    std::ofstream file(file_name, std::ios::binary);
    pixel_generator generator(m_pixels, m_width, m_height);
    generator.write(file);
}
#endif
