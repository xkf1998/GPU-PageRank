import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import java.io.IOException;

public class PageRankMapper extends Mapper<LongWritable, Text, Text, NodePRTuple> {
    @Override
    protected void map(LongWritable key, Text value, Mapper<LongWritable, Text, Text, NodePRTuple>.Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] arr = line.split(" ");
        String src = arr[0];
        Double initPR = Double.parseDouble(arr[arr.length-1]);
        Integer numberOfTarget = arr.length - 2;
        Double avgPR = initPR/numberOfTarget;
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i < arr.length-1; ++i) {
            // C: A,PR
            String target = arr[i];
            sb.append(target).append(' ');
            NodePRTuple tp = new NodePRTuple(src, avgPR);
            context.write(new Text(target), tp);
        }
        sb.setLength(sb.length()-1);
        // A: C J
        NodePRTuple tp = new NodePRTuple(sb.toString());
        context.write(new Text(src), tp);
    }
}
