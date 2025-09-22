import java.io.IOException;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;

public class CheckpointIO {
    public static class Config {
        public int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len;
    }

    public static Config readConfig(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer buf = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size()).order(ByteOrder.LITTLE_ENDIAN);
            Config cfg = new Config();

            cfg.dim = buf.getInt();
            cfg.hidden_dim = buf.getInt();
            cfg.n_layers = buf.getInt();
            cfg.n_heads = buf.getInt();
            cfg.n_kv_heads = buf.getInt();
            cfg.vocab_size = buf.getInt();
            cfg.max_seq_len = buf.getInt();
            return cfg;
        }
    }

    public static FloatBuffer readWeights(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer file = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size()).order(ByteOrder.LITTLE_ENDIAN);
            file.position(28);
            return file.slice().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        }
    }
}
