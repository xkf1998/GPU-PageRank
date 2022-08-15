import org.apache.hadoop.io.Writable;
import javax.annotation.Nullable;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class NodePRTuple implements Writable {
    private String node = null;
    public NodePRTuple() {}
    @Nullable
    private String pr = null;
    public NodePRTuple(String node, Double pr) {
        this.node = node;
        this.pr = pr.toString();
    }

    public NodePRTuple(String nodes) {
        node = nodes;
        pr = null;
    }

    public String getNode() {
        return this.node;
    }

    @Nullable
    public String getPr() {
        return this.pr;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        dataOutput.writeBytes(node);
        if (pr != null) {
            dataOutput.writeBytes(",");
            dataOutput.writeBytes(pr);
        }
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        String in = dataInput.readLine();
        String[] arr = in.split(",");
        if (arr.length == 2) {
            node = arr[0];
            pr = arr[1];
        } else {
            node = arr[0];
            pr = null;
        }
    }
}

