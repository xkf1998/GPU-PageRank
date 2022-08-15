import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;


public class PageRank {
    private static Path getOutputPath(final int i, final String originPath) {
        return new Path(originPath + i);
    }
    public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
        if (args.length != 3) {
            System.err.println("Usage: PageRank <input> <output path> <iterations>");
            System.exit(-1);
        }

        // preprocess
        Path inputPath = new Path(args[0]);
        Path outputPath = new Path(args[1]);
        int iterations = Integer.parseInt(args[2]);
        Job preprocessJob = Job.getInstance();
        preprocessJob.setJarByClass(PageRank.class);
        preprocessJob.setJobName("Preprocess");
        preprocessJob.setMapperClass(PreprocessMapper.class);
        preprocessJob.setReducerClass(PreprocessReducer.class);
        preprocessJob.setOutputKeyClass(Text.class);
        preprocessJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(preprocessJob, inputPath);
        FileOutputFormat.setOutputPath(preprocessJob, outputPath);
        preprocessJob.waitForCompletion(true);
        inputPath = outputPath;

        // page rank
        for (int i = 0; i < iterations; ++i) {
            outputPath = getOutputPath(i, args[1]);
            Job job = Job.getInstance();
            job.setNumReduceTasks(1);
            job.setJarByClass(PageRank.class);
            job.setJobName("Page Rank Simulation");
            job.setMapperClass(PageRankMapper.class);
            job.setReducerClass(PageRankReducer.class);

            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(NodePRTuple.class);
            FileInputFormat.addInputPath(job, inputPath);
            FileOutputFormat.setOutputPath(job, outputPath);
            job.waitForCompletion(true);
            inputPath = outputPath;
        }
    }
}
