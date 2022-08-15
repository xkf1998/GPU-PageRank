import org.apache.log4j.Logger;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import java.io.IOException;

public class PageRankReducer extends Reducer<Text, NodePRTuple, Text, Text> {
    static final double dampingFactor = 0.85;
    @Override
    protected void reduce(Text key, Iterable<NodePRTuple> values, Reducer<Text, NodePRTuple, Text, Text>.Context context) throws IOException, InterruptedException {
        double totalPR = 1 - dampingFactor;
        StringBuilder outputValue = new StringBuilder();
        for (NodePRTuple v: values) {
            String prStr = v.getPr();
            if (prStr == null) {
                String node = v.getNode();
                outputValue.append(key.toString()).append(" ").append(node);
                continue;
            }
            double pr = Double.parseDouble(prStr);
            totalPR += pr;
        }
        totalPR = totalPR * dampingFactor;
        if (outputValue.length() != 0) {
            outputValue.append(" ").append(totalPR);
            context.write(null, new Text(outputValue.toString()));
        } else {
            // key is only used as in-node not as out-node
            outputValue.append(key.toString()).append(" ").append(totalPR);
        }
    }
}
