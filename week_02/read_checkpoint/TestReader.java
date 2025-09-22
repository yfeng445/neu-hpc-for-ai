import java.nio.*;

public class TestReader {
    public static void main(String[] args) throws Exception {
        // point to the local file in the same folder
        String modelPath = "./stories15M.bin";

        CheckpointIO.Config cfg = CheckpointIO.readConfig(modelPath);
        System.out.println("Config:");
        System.out.printf("dim=%d, hidden=%d, layers=%d, heads=%d, kv_heads=%d, vocab=%d, seq_len=%d%n",
            cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.max_seq_len);

        FloatBuffer fb = CheckpointIO.readWeights(modelPath);
        System.out.println("First 10 floats:");
        for (int i = 0; i < 10 && fb.hasRemaining(); i++) {
            System.out.print(fb.get() + " ");
        }
        System.out.println();
    }
}
