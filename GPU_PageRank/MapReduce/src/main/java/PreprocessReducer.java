import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class PreprocessReducer extends Reducer<Text, Text, Text, Text> {
    static final double initPR = 0.85;
    @Override
    protected void reduce(Text key, Iterable<Text> values, Reducer<Text, Text, Text, Text>.Context context) throws IOException, InterruptedException {
        StringBuilder sb = new StringBuilder();
        sb.append(key.toString()).append(" ");
        for (Text value : values) {
            sb.append(value.toString()).append(" ");
        }
        sb.append(initPR);
        context.write(null, new Text(sb.toString()));
    }
}