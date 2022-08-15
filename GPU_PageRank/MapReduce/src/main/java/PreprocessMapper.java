import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class PreprocessMapper extends Mapper<LongWritable, Text, Text, Text> {
    @Override
    protected void map(LongWritable key, Text value, Mapper<LongWritable, Text, Text, Text>.Context context) throws IOException, InterruptedException {
        if (key.get() == 0) {
            return;
        } else {
            String[] tokens = value.toString().split(" ");
            String to = tokens[0];
            String from = tokens[1];
            context.write(new Text(from), new Text(to));
        }
    }
}