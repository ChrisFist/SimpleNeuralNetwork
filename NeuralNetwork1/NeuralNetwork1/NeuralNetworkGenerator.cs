using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Binding;


namespace NeuralNetwork1
{
    public class NeuralNetworkGenerator
    {

        public Graph BuildNeuralNetwork()
        {
            var graph = new Graph();

            // Define the input layer
            var input = tf.placeholder(TF_DataType.TF_FLOAT, new Shape(-1, 7));

            // Define the hidden layers
            var hidden1 = tf.add(tf.matmul(input, tf.constant(new float[7, 10])), tf.constant(new float[10]));
            var hidden2 = tf.add(tf.matmul(hidden1, tf.constant(new float[10, 10])), tf.constant(new float[10]));

            // Define the output layer
            var output = tf.add(tf.matmul(hidden2, tf.constant(new float[10, 1])), tf.constant(new float[1]));

            // Apply the activation function to the output layer (e.g., sigmoid)
            var activatedOutput = tf.sigmoid(output);

            return graph;
        }

        public void TrainNeuralNetwork(Graph graph, double[][] trainingData, int numEpochs)
        {
            // Prepare the training data
            var inputs = trainingData.Select(data => data.Take(data.Length - 1).ToArray()).ToArray();
            var labels = trainingData.Select(data => data[data.Length - 1]).ToArray();

            // Create a session
            using (var session = tf.Session(graph))
            {
                // Initialize variables
                session.run(tf.global_variables_initializer());

                // Define placeholders for inputs and labels
                var inputTensor = graph.get_operation_by_name("input").outputs[0];
                var labelTensor = graph.get_operation_by_name("label").outputs[0];

                // Define the loss and optimizer
                var outputTensor = graph.get_operation_by_name("activated_output").outputs[0];
                var labelPlaceholder = tf.placeholder(TF_DataType.TF_FLOAT, new Shape(-1));
                var loss = tf.reduce_mean(tf.square(tf.subtract(labelPlaceholder, outputTensor)));
                var optimizer = tf.train.GradientDescentOptimizer(learning_rate: 0.01f).minimize(loss);

                // Perform training
                for (int epoch = 0; epoch < numEpochs; epoch++)
                {
                    // Run one training step
                    session.run(optimizer, new FeedItem(inputTensor, inputs), new FeedItem(labelPlaceholder, labels));
                }
            }
        }
    }
}
