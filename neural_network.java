/*
 * Name: Abigail Folds
 * ID: 10395231
 * Date: October 24, 2023
 * Assignment 2
 * Description: This program implements a sigmoid neuron feed-forward neural network.
 * It provides a end-user menu that allows you to train the network, svae your results to
 * a file and allows you to load your own weights into the network. You can also see how
 * well a network does under testing and training data, along will having the network
 * go image by image a A) do all of them or B) only how the images of the tests it got 
 * incorrect.
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import java.io.FileWriter;
import java.text.DecimalFormat;

public class neural_network{

    public static void main(String[] args){ 
        // I dont want this stuff getting wiped by the loop.
        // All of the Variables that are used by multiple cases, and
        // need to be used between cases, and not local to that case.
        String fileName;
        int epoch = 1; //default
        double data[][];
        double tests[][];
        double weightsLevel1[][] = null;
        double weightsLevel2[][] = null;
        double biasLevel1[][] = null;
        double biasLevel2[][] = null;

        // There used to be another variable here, for testing and now
        // this one is the only one left, its for calculating the Accuracy
        double newAcc = 0;

        Scanner scanner = new Scanner(System.in);  // Create a Scanner object
        DecimalFormat df = new DecimalFormat("#.####"); // Pretty output

        // Main loop for network
        while(true){
            // Printing the menu
            printMenu();
            // Getting the input
            int input = scanner.nextInt();  // Read user input
            System.out.println();

            // Case system for output!
            if(input == 0) { // Exit the program
                scanner.close();
                System.exit(0);
            } else if(input == 1) { // Run the default trainer!
                // TRAINING DATAAAAAAA
                // Getting inputs from a file
                fileName = "mnist_train.csv"; //name of file
                double fileInput[][] = readCSV(fileName);

                // Constants! (Dont forget the ETA)
                int eta = 3;
                epoch = 30;
                int testInMini = 10;
                int numMiniBatch = (fileInput[0].length)/testInMini;
                        
                // Inputs and Outputs
                //
                //// Here we get a matrix of just the correct output and testing data
                //// along with a matrix of some randomized indexes for testing in batches.
                data = dataGet(fileInput);
                tests = testsGet(fileInput);
                int miniBatchIndexes[] = fisherYatesShuffle(tests);

                //// Starting Weights and Biases
                //// These are random numbers between [-1, 1]
                weightsLevel1 = initializeMatrix(new double[15][784]);
                weightsLevel2 = initializeMatrix(new double[10][15]);

                biasLevel1 = initializeMatrix(new double[15][1]);
                biasLevel2 = initializeMatrix(new double[10][1]);

                // Actual Training of the network below!

                // This loop is to repeat until the epoch is met
                for(int k = 0; k < epoch; k++){

                    // The current minibatch
                    int currentBatch = 0;

                    int rightanswers[] = new int[10]; // Sum array of # times the network correctly identified a number
                    int timesShown[] = new int[10]; // # of times a number showed up

                    // This loop is to repeat until all minibatches are complete
                    for(int j = 0; j < numMiniBatch; j++){
                        // Getting the data for the current minibatch using the indexes from the current
                        // batch as an offset,
                        double batchOutput[][] = new double[data.length][testInMini]; //Correct output
                        double batchInput[][] = new double[tests.length][testInMini]; //What the network is looking at
                        
                        for(int i = 0; i<testInMini; i++){
                            for(int n = 0; n< batchOutput.length; n++){
                                batchOutput[n][i] = data[n][miniBatchIndexes[i+currentBatch]];
                            }
                            for(int n = 0; n< batchInput.length; n++){
                                batchInput[n][i] = tests[n][miniBatchIndexes[i+currentBatch]];
                            }
                        }

                        //gradient sums, so i can reuse the variables 
                        double sumGbLevel1[][] = new double[biasLevel1.length][biasLevel1[0].length];
                        double sumGbLevel2[][] = new double[biasLevel2.length][biasLevel2[0].length];

                        double sumGwLevel1[][] = new double[weightsLevel1.length][weightsLevel1[0].length];
                        double sumGwLevel2[][] = new double[weightsLevel2.length][weightsLevel2[0].length]; 

                        // This loop is to repeat for every test in a mini-batch
                        for(int i = 0; i<testInMini; i++){
                            double forwardpass1[][] = Sigmoid(MatrixAddition(MatrixMul(weightsLevel1, ExtractMatrix(batchInput, i)),biasLevel1));
                            double forwardpass2[][] = Sigmoid(MatrixAddition(MatrixMul(weightsLevel2, forwardpass1), biasLevel2));

                            double gbLevel2[][] = gradientBiasLayer2(forwardpass2, ExtractMatrix(batchOutput, i));
                            double gwLevel2[][] = gradientWeights(forwardpass1, gbLevel2);

                            double gbLevel1[][] = gradientBiasLayer1(weightsLevel2, gbLevel2, forwardpass1);
                            double gwLevel1[][] = gradientWeights(ExtractMatrix(batchInput,i), gbLevel1);

                            // Sums for calculating the new weights
                            sumGbLevel1 = MatrixAddition(sumGbLevel1, gbLevel1);
                            sumGbLevel2 = MatrixAddition(sumGbLevel2, gbLevel2);
                            sumGwLevel1 = MatrixAddition(sumGwLevel1, gwLevel1);
                            sumGwLevel2 = MatrixAddition(sumGwLevel2, gwLevel2);

                            //Here we are going to find if the network was right
                            int indexFP = getMax(forwardpass2);
                            int indexD = getMax(ExtractMatrix(batchOutput, i));
                            // checking if the output from this test is correct or not.
                            if(indexD == indexFP){
                                rightanswers[indexD] = rightanswers[indexD] + 1;
                            }

                            timesShown[indexD] = timesShown[indexD] + 1;

                        }

                            // Now we need to compute the new Bias and Weights
                            weightsLevel1 = revisedWeights(sumGwLevel1, weightsLevel1, eta);
                            weightsLevel2 = revisedWeights(sumGwLevel2, weightsLevel2, eta);

                            biasLevel1 = revisedBias(sumGbLevel1, biasLevel1, eta);
                            biasLevel2 = revisedBias(sumGbLevel2, biasLevel2, eta);

                            // Now we move on to the next batch moving the offset to find the next indexes for minibatchindexes. 
                            currentBatch+=testInMini;
            
                    }
                    System.out.println("epoch " +k);
                    int allRight = 0;
                    int testNum = 0;
                    for(int i=0; i<timesShown.length;i++){
                        System.out.print(i + ": " + rightanswers[i] + "/" + timesShown[i] + " ");
                        allRight += rightanswers[i];
                        testNum += timesShown[i];
                    }
                    newAcc = ((double)allRight/testNum)*100;
                    System.out.println("Accuracy = " + allRight +"/"+testNum + " = " + df.format(newAcc) );
                    System.out.println();

                }

            } else if(input == 2) { // Load weights and biases in
                weightsLevel1 = reverseMatrix(readCSV("weightsLevel1.csv"));
                weightsLevel2 = reverseMatrix(readCSV("weightsLevel2.csv"));
                biasLevel1 = reverseMatrix(readCSV("biasLevel1.csv"));
                biasLevel2 = reverseMatrix(readCSV("biasLevel2.csv"));
            } else if(input == 3) { // Display acc on TRAIN data
                if(weightsLevel1 != null & weightsLevel2 != null 
                    & biasLevel1 != null & biasLevel2 != null){ // This makes sure that there are loaded weights and biases
                    
                    // TRAINING DATAAAAAAA
                    // Getting inputs from a file
                    fileName = "mnist_train.csv"; //name of file
                    double fileInput[][] = readCSV(fileName);
                            
                    // Inputs and Outputs
                    //
                    //// Here we get a matrix of just the correct output and testing data
                    //// along with a matrix of some randomized indexes for testing in batches.
                    data = dataGet(fileInput);
                    tests = testsGet(fileInput);

                    int rightanswers[] = new int[10]; // Sum array of # times the network correctly identified a number
                    int timesShown[] = new int[10]; // # of times a number showed up

                    // This loop is to repeat for every test in a mini-batch
                    for(int i = 0; i<tests[0].length; i++){
                        double forwardpass1[][] = Sigmoid(MatrixAddition(MatrixMul(weightsLevel1, ExtractMatrix(tests, i)),biasLevel1));
                        double forwardpass2[][] = Sigmoid(MatrixAddition(MatrixMul(weightsLevel2, forwardpass1), biasLevel2));

                        //Here we are going to find if the network was right
                        int indexFP = getMax(forwardpass2);
                        int indexD = getMax(ExtractMatrix(data, i));
                        // checking if the output from this test is correct or not.
                        if(indexD == indexFP){
                            rightanswers[indexD] = rightanswers[indexD] + 1;
                        }

                        timesShown[indexD] = timesShown[indexD] + 1;
                    }
    
                    System.out.println("Training Output");
                    int allRight = 0;
                    int testNum = 0;
                    for(int i=0; i<timesShown.length;i++){
                        System.out.print(i + ": " + rightanswers[i] + "/" + timesShown[i] + " ");
                        allRight += rightanswers[i];
                        testNum += timesShown[i];
                    }
                    newAcc = ((double)allRight/testNum)*100;
                    System.out.println("Accuracy = " + allRight +"/"+testNum + " = " + df.format(newAcc) );
                    System.out.println();
                } else {
                    System.out.println("Insuffiecient input, please use [1] or [2] first!");
                    System.out.println();
                }
            } else if(input == 4) { // Display acc on TESTING data
                if(weightsLevel1 != null & weightsLevel2 != null 
                    & biasLevel1 != null & biasLevel2 != null){ // This makes sure that there are loaded weights and biases
        
                    // TESTING DATAAAAAAA
                    // Getting inputs from a file
                    fileName = "mnist_test.csv"; //name of file
                    double fileInput[][] = readCSV(fileName);
                            
                    // Inputs and Outputs
                    //
                    //// Here we get a matrix of just the correct output and testing data
                    //// along with a matrix of some randomized indexes for testing in batches.
                    data = dataGet(fileInput);
                    tests = testsGet(fileInput);

                    // Actual Testing of the Network below!
                    int rightanswers[] = new int[10]; // Sum array of # times the network correctly identified a number
                    int timesShown[] = new int[10]; // # of times a number showed up

                    // This loop is to repeat for every test in a mini-batch
                    for(int i = 0; i<tests[0].length; i++){
                        double forwardpass1[][] = Sigmoid(MatrixAddition(MatrixMul(weightsLevel1, ExtractMatrix(tests, i)),biasLevel1));
                        double forwardpass2[][] = Sigmoid(MatrixAddition(MatrixMul(weightsLevel2, forwardpass1), biasLevel2));

                        //Here we are going to find if the network was right
                        int indexFP = getMax(forwardpass2);
                        int indexD = getMax(ExtractMatrix(data, i));
                        // checking if the output from this test is correct or not.
                        if(indexD == indexFP){
                            rightanswers[indexD] = rightanswers[indexD] + 1;
                        }

                        timesShown[indexD] = timesShown[indexD] + 1;
                    }  

                    System.out.println("Testing Output");
                    int allRight = 0;
                    int testNum = 0;
                    for(int i=0; i<timesShown.length;i++){
                        System.out.print(i + ": " + rightanswers[i] + "/" + timesShown[i] + " ");
                        allRight += rightanswers[i];
                        testNum += timesShown[i];
                    }
                    newAcc = ((double)allRight/testNum)*100;
                    System.out.println("Accuracy = " + allRight +"/"+testNum + " = " + df.format(newAcc) );
                    System.out.println();
                } else {
                    System.out.println("Insuffiecient input, please use [1] or [2] first!");
                    System.out.println();
                }
            } else if(input == 5) { // Run network on TESTING data with IMAGES
                if(weightsLevel1 != null & weightsLevel2 != null 
                    & biasLevel1 != null & biasLevel2 != null){ // This makes sure that there are loaded weights and biases
        
                    // TESTING DATAAAAAAA
                    // Getting inputs from a file
                    fileName = "mnist_test.csv"; //name of file
                    double fileInput[][] = readCSV(fileName);

                    int cont = 1;
                            
                    // Inputs and Outputs
                    //
                    //// Here we get a matrix of just the correct output and testing data
                    //// along with a matrix of some randomized indexes for testing in batches.
                    data = dataGet(fileInput);
                    tests = testsGet(fileInput);

                    // This loop is to repeat for every test in a mini-batch
                    int i = 0;
                    while(cont == 1 | i==tests[0].length){
                        double forwardpass1[][] = Sigmoid(MatrixAddition(MatrixMul(weightsLevel1, ExtractMatrix(tests, i)),biasLevel1));
                        double forwardpass2[][] = Sigmoid(MatrixAddition(MatrixMul(weightsLevel2, forwardpass1), biasLevel2));

                        //Here we are going to find if the network was right
                        int maxIndexN = getMax(forwardpass2);
                        double maxIndexC = getMax(ExtractMatrix(data, i));
                        String correct = "Incorrect."; //Assume incorrect

                        // checking if the output from this test is correct or not.
                        if(data[maxIndexN][i] == 1.0){
                            correct = "Correct.";
                        }

                        // Printing output
                        System.out.println("Testing Case #"+(i+1)+":  Correct classification = " 
                            + maxIndexC +"  Network Output = "+maxIndexN+ "  " +correct);
                        printAscii(ExtractMatrix(tests, i));
                        System.out.println();
                        System.out.println("Enter 1 to continue. All other values to main menu.");
                        cont = scanner.nextInt();
                        i++;
                    } 

                } else {
                    System.out.println("Insuffiecient input, please use [1] or [2] first!");
                    System.out.println();
                }
            } else if(input == 6) { // Run network on TESTING data with IMAGES but only incorrect
                if(weightsLevel1 != null & weightsLevel2 != null 
                    & biasLevel1 != null & biasLevel2 != null){ // This makes sure that there are loaded weights and biases
        
                    // TESTING DATAAAAAAA
                    // Getting inputs from a file
                    fileName = "mnist_test.csv"; //name of file
                    double fileInput[][] = readCSV(fileName);

                    int cont = 1;
                            
                    // Inputs and Outputs
                    //
                    //// Here we get a matrix of just the correct output and testing data
                    //// along with a matrix of some randomized indexes for testing in batches.
                    data = dataGet(fileInput);
                    tests = testsGet(fileInput);

                    // This loop is to repeat for every test in a mini-batch
                    int i = 0;
                    while(cont == 1 | i==tests[0].length){
                        double forwardpass1[][] = Sigmoid(MatrixAddition(MatrixMul(weightsLevel1, ExtractMatrix(tests, i)),biasLevel1));
                        double forwardpass2[][] = Sigmoid(MatrixAddition(MatrixMul(weightsLevel2, forwardpass1), biasLevel2));

                        //Here we are going to find if the network was right
                        int maxIndexN = getMax(forwardpass2);
                        double maxIndexC = getMax(ExtractMatrix(data, i));
                        String correct = "Incorrect."; //Assume incorrect

                        // checking if the output from this test is correct or not.
                        if(data[maxIndexN][i] == 1.0){
                            correct = "Correct.";
                        }

                        // Printing output only if incorrect
                        if(correct == "Incorrect."){
                            System.out.println("Testing Case #"+(i+1)+":  Correct classification = " 
                                + maxIndexC +"  Network Output = "+maxIndexN+ "  " +correct);
                            printAscii(ExtractMatrix(tests, i));
                            System.out.println();
                            System.out.println("Enter 1 to continue. All other values to main menu.");
                            cont = scanner.nextInt();
                        }
                        i++;
                    } 

                } else {
                    System.out.println("Insuffiecient input, please use [1] or [2] first!");
                    System.out.println();
                }
            } else if(input == 7) { // Save the network state to file
                if(weightsLevel1 != null & weightsLevel2 != null 
                    & biasLevel1 != null & biasLevel2 != null){ // This makes sure that there are loaded weights and biases
                    try {
                        writeCSV(weightsLevel1, "weightsLevel1.csv");
                        writeCSV(weightsLevel2, "weightsLevel2.csv");
                        writeCSV(biasLevel1, "biasLevel1.csv");
                        writeCSV(biasLevel2, "biasLevel2.csv");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    System.out.println("Insuffiecient input, please use [1] or [2] first!");
                    System.out.println();
                }
            } else {
                ; // This is the pass equiv
            }

        }
        
    }

    // This method allows us to read data from a CSV file and puts it into one double 2darray.
    static double[][] readCSV(String filePath) {
        // initialization
        double[][] images = null;

        // This try-catch will make sure the file accually exsists. Because we run through the
        // file once to get the size, I chose to close and open it again, rather than having 
        // another try-catch. 
        try (BufferedReader br = new BufferedReader(new FileReader(filePath)) ) {
            String line;
            int rowCount = 0;
            int colCount = 0;

            // Counting rows and cols
            while ((line = br.readLine()) != null) {
                rowCount++;
                String[] rowValues = line.split(",");
                colCount = Math.max(colCount, rowValues.length);
            }

            // Making the array to return
            images = new double[colCount][rowCount];
            
            // Open and close to reset the position to the top of the file.
            // I use a different variable name because there were some errors
            // and I did not want to make two try catch
            br.close();
            BufferedReader pr = new BufferedReader(new FileReader(filePath)); // new reader

            // Gathering the actual data from the file, making sure that everything is turned 
            // into a double, This array is 60000 x 785 at max.
            int col = 0;
            while ((line = pr.readLine()) != null) {
                String[] rowValues = line.split(",");
                for (int row = 0; row < rowValues.length; row++) {
                    images[row][col] = Double.parseDouble(rowValues[row]);                    
                }
                col++;
            }
            // closing the file
            pr.close();
        // just in case
        } catch (IOException e) {
            e.printStackTrace();
        }

        return images;
    }

    // This method allows us to write to a CSV file from one double 2darray
    static void writeCSV(double[][] matrix, String fileName) throws IOException{

        try(FileWriter writer = new FileWriter(fileName)){

            // This is going through the entire matrix
            for(int row = 0; row<matrix.length; row++){
                for(int col = 0; col<matrix[0].length; col++){
                    // writing the element into the file and adding a ',' after
                    writer.append(String.valueOf(matrix[row][col]));
                    if(col < matrix[0].length-1) {
                        writer.append(",");
                    }
                }
                //end of col
                writer.append("\n");
            }
        }

    }

    // Method for printing out a Matrix for testing purposes.
    static void printMatrix(double M[][]){

        // This is just to allow the testing output to be a bit easier to read and look like the 
        // output in the excel.
        DecimalFormat df = new DecimalFormat("#.####");

        for(int i = 0; i < M.length; i++){
            for(int j = 0; j < M[0].length; j++){
                System.out.print(df.format(M[i][j])+" "); 
            }
            System.out.println();
        }
    }

    // This method initalizes a Matrix of any size, with random number between -1 and 1
    static double[][] initializeMatrix(double[][] inputMatrix){
        Random random = new Random(); // my random number

        // just filling in the entire predetermined array with random numbers in [-1,1]
        for(int i=0; i<inputMatrix.length; i++){
            for(int j=0; j<inputMatrix[0].length;j++){
                //This originaly only gets numbers [0,1] but this equation allows it to be [-1,1]
                inputMatrix[i][j] = random.nextDouble() * 2 - 1; 
            }
        }
        return inputMatrix;
    }

    // Method to extract a nxm Matrix from a nxm Matrix. These are the holding martrcies.
    // There is a optional param here, most of the time this function is used on a nx1 matrix, 
    // but there is a point where the Working Batch data needs to be changed, so thats allowing you to 
    // select an ending index as well.
    static double[][] ExtractMatrix(double A[][], int Begindex, int... endIndex){
        double C[][];

        // This is the function for if there is no value in the optional Parameter
        // It grab all the items from a certain column [index] to create a nx1 matrix
        if(endIndex.length == 0){
            C = new double[A.length][1];
            for(int i = 0; i<C.length; i++){
                C[i][0] = A[i][Begindex];
            }
        } else { //This function if for gathering a larger set of data from multiple indexes,
            // They must be next to each other.
            C = new double[A.length][2];
            for(int i = 0; i<C.length; i++){
                for(int j = 0; j <= endIndex[0]-Begindex; j++){
                    C[i][j] = A[i][j+Begindex];
                }
            }
        }

        return C;
    }

    // Method for matrix Multiplication
    static double[][] MatrixMul(double A[][], double B[][]){

        // Error Checking
        if (A[0].length != B.length){
            System.out.println("Multiplication not possible");
            return null;
        }
        double C[][] = new double[A.length][B[0].length];

        for (int i = 0; i < A.length; i++){
            for (int j = 0; j < B[0].length; j++){
                for (int k=0; k < B.length; k++){
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return C;
    }

    // Method for Matrix Addition
    static double[][] MatrixAddition(double A[][], double B[][]){

        // Error Checking
        if (A.length != B.length || A[0].length != B[0].length){
            System.out.println("Addtion not possible");
            return null;
        }
        double C[][] = new double[A.length][A[0].length];

        for(int i = 0; i<C.length; i++){
            for(int j = 0; j<C[0].length; j++){
                C[i][j] = A[i][j] + B[i][j];
            }
        }

        return C;
    }
    
    // Method Matrix Scalar Multiplication. This is never used, I just implemented it
    // because I was going through implementing all matrix math
    static double[][] MatrixScalar(double A[][], double Scalar){
        double C[][] = new double[A.length][A[0].length];

        for(int i = 0; i<C.length;i++){
            for(int j = 0; j<C[0].length; j++){
                C[i][j] = A[i][j] * Scalar;
            }
        }

        return C;
    }

    // Method for the Sigmoid Function
    // This function uses Math.pow to compute the power of e^(A_ij)
    static double[][] Sigmoid(double A[][]){
        double C[][] = new double[A.length][A[0].length];

        // for e i used 2.71828
        for(int i = 0; i<C.length;i++){
            for(int j = 0; j<C[0].length;j++){
                C[i][j] = 1/(1+Math.pow(2.71828,(-A[i][j])));
            }
        }

        return C;
    }

    // Method that computes the bias gradient for Backward pass from Layer 3 -> Layer 2
    static double[][] gradientBiasLayer2(double activation[][], double train[][]){
        double gb[][] = new double[activation.length][activation[0].length];

        for(int i = 0; i< gb.length;i++){
            for(int j=0; j<gb[0].length;j++){
                gb[i][j] = (activation[i][j] - train[i][j]) * activation[i][j] * (1-activation[i][j]);
            }
        }
        return gb;
    }

    // Method that computes the bias gradient for Backward pass form Layer 2 -> Layer 1
    static double[][] gradientBiasLayer1(double weights[][], double oldgb[][], double activation[][]){
        double gb[][] = new double[weights[0].length][1];

        for(int i = 0; i<gb.length; i++){
            double sum = 0;
            for(int j = 0; j<weights.length; j++){
                sum += weights[j][i]*oldgb[j][0];
            }
            gb[i][0] = sum * (activation[i][0] * (1-activation[i][0]));
        }
        return gb;
    }

    // Method that computes the weight gradient. Works will all layers
    static double[][] gradientWeights(double activation[][], double gb[][]){
        double gw[][] = new double[gb.length][activation.length];

        for(int i=0; i<gw.length; i++){
            for(int j=0; j<gw[0].length; j++){
                gw[i][j] = gb[i][0]*activation[j][0];
            }
        }
        return gw;
    }

    // Method that calculate the revised biases
    static double[][] revisedBias(double sumBiasG[][], double prevBias[][], int eta ){
        double newBias[][] = new double[prevBias.length][1];
        
        for(int i=0; i<newBias.length; i++){
            newBias[i][0] = (prevBias[i][0])-(eta/2)*(sumBiasG[i][0]);
        }
        return newBias;
    }

    // Method that calculates the revised weights
    static double[][] revisedWeights(double sumWeightG[][], double prevWeight[][], int eta ){
        double newWeight[][] = new double[prevWeight.length][prevWeight[0].length];

        for(int i=0; i<newWeight.length; i++){
            for(int j=0; j<newWeight[0].length; j++){
                newWeight[i][j] = (prevWeight[i][j])-(eta/2)*(sumWeightG[i][j]);
            }
        }
        return newWeight;
    }

    // This translates a number into its one hot vector equivalent
    static double[][] translateToHotVector(double number) {
        double hotVector[][] = new double[10][1];

        hotVector[(int)number][0]=1;
        return hotVector;
    }

    // This method will take in our input data and create a new array with ONLY the answers.
    static double[][] dataGet(double[][] fileInput){
        double data[][] = new double[10][fileInput[0].length];

        for(int i = 0; i<data[0].length; i++){
            double hotVector[][] = translateToHotVector(fileInput[0][i]);
            for(int j = 0; j<hotVector.length; j++){
                data[j][i] = hotVector[j][0];
            }
        }
        return data;
    }
    
    // This method will take in our input data and create a new array with ONLY the tests.
    static double[][] testsGet(double[][] fileInput){
        double tests[][] = new double[784][fileInput[0].length];

        for(int i=0; i<tests[0].length; i++){
            for(int j = 0; j<tests.length; j++){
                tests[j][i] = fileInput[j+1][i]/255.0;
            }
        }

        return tests;
    }

    // This is using the Fisher-Yates Shuffle algorithm!
    // I am using the sudo-code from this page of the modern algorithm
    // https://en.wikipedia.org/wiki/Fisher–Yates_shuffle
    // This algorithm has also been merged with the only to make the list of
    // indicies!
    static int[] fisherYatesShuffle(double[][] arrayIn){
        int indexes[] = new int[arrayIn[0].length];
        Random random = new Random();

        // Getting the indeicies here.
        for(int i = 0; i<indexes.length; i++){
            indexes[i] = i;
        }

        // I am using the first example of the algorithm
        /* (Sudo-code from website)
        -- To shuffle an array a of n elements (indices 0..n-1):
            for i from n−1 down to 1 do
                j ← random integer such that 0 ≤ j ≤ i
                exchange a[j] and a[i]
        */
        for(int i = indexes.length-1; i>0; i--){
            int j = random.nextInt(i);
            int temp1 = indexes[i];
            int temp2 = indexes[j];
            indexes[j] = temp1;
            indexes[i] = temp2;
        }
        return indexes;
    }

    // This is the function that prints out the menu
    static void printMenu(){
        System.out.println("Welcome to the Menu!");
        System.out.println("[1]: Train the network");
        System.out.println("[2]: Load a pre-trained network");
        System.out.println("[3]: Display network accuracy on TRAINING data");
        System.out.println("[4]: Display network accuracy on TESTING data");
        System.out.println("[5]: Run network on TESTING data showing images and labels.");
        System.out.println("[6]: Display the misclassified TESTING images");
        System.out.println("[7]: Save the network state to file");
        System.out.println("[0]: Exit");
        System.out.println();
    }

    // This method will reverse a function. (for reading weight set from a CSV);
    static double[][] reverseMatrix(double[][] arrayIn){
        double arrayOut[][] = new double[arrayIn[0].length][arrayIn.length];

        for(int row = 0; row<arrayOut.length; row++){
            for(int col = 0; col<arrayOut[0].length; col++){
                arrayOut[row][col] = arrayIn[col][row];
            }
        }

        return arrayOut;
    }

    // This method will print out the ascii image of a 784x1 array
    static void printAscii(double[][] arrayIn){

        // my ascii characters (11)
        char[] asciiChars = { ' ', '.', ':', '-', '=', '+', '*', '#', '%', '8', '@' };

        // i is the row and j is the col.
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                // the *255 is because when the file is read i divide by 255, 
                // The type of character is based on how "Light" the pixel is
                // i.e. pixels closer to 255 are darker. 
                double pixelValue = arrayIn[i * 28 + j][0] * 255; 
                int index = (int) (pixelValue / 25.5); // Map pixel value to index 255/25.5 = 10
                System.out.print(asciiChars[index]);
            }
            System.out.println(); // Move to the next line for the next row
        }
    }

    // This method finds the index of the max number in a array
    // Array assumed to be a nx1 matrix
    static int getMax(double[][] arrayIn){
        int maxIndex = 0; //max assumed to be first element
        for(int n=1;n<arrayIn.length; n++){
            if(arrayIn[n][0] > arrayIn[maxIndex][0]){
                maxIndex = n;
            }
        }
        return maxIndex;
    }


}