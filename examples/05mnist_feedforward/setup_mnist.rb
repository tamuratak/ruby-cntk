require 'zlib'

# Based on http://d.hatena.ne.jp/n_shuyo/20090913/mnist
# MIT / 2-Clause BSD License

def read_mnist_images(filename)
  images = []
  Zlib::GzipReader.open(filename) do |f|
    magic, n_images = f.read(8).unpack('N2')
    raise 'This is not MNIST image file' if magic != 2051
    n_rows, n_cols = f.read(8).unpack('N2')
    n_images.times do
      images << f.read(n_rows * n_cols)
    end
  end
  images
end

def read_mnist_label(filename)
  labels = []
  Zlib::GzipReader.open(filename) do |f|
    magic, n_labels = f.read(8).unpack('N2')
    raise 'This is not MNIST label file' if magic != 2049
    labels = f.read(n_labels).unpack('C*')
  end
  labels
end

def save_to_txtfile(filename, images, labels)
  File.open(filename, 'w'){|io|
    (1..labels.size).each{|idx|
      idx0 = idx - 1
      label = [0] * 10
      label[ labels[idx0] ] = 1
      io.puts "#{idx}|x " + images[idx0].unpack('C*').map{|i| "%3d" % i }.join(" ") + " |y #{label.join(' ')}"
    }
  }
end

#images[0].unpack('C*').each_slice(28){|a|
#puts a.map{|i| "%3d" % i }.join(" ")
#}

images = read_mnist_images('train-images-idx3-ubyte.gz')
labels = read_mnist_label( 'train-labels-idx1-ubyte.gz')
save_to_txtfile('mnist_train.txt', images, labels)

images = read_mnist_images('t10k-images-idx3-ubyte.gz')
labels = read_mnist_label( 't10k-labels-idx1-ubyte.gz')
save_to_txtfile('mnist_test.txt', images, labels)
