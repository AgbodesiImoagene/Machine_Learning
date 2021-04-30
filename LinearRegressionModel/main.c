#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define STR_BUFFER 2048
#define TRAINING_INPUT_FILE "trainingInput"
#define TRAINING_OUTPUT_FILE "trainingOutput"
#define CV_INPUT_FILE "CVInput"
#define CV_OUTPUT_FILE "CVOutput"
#define TEST_INPUT_FILE "testInput"
#define TEST_OUTPUT_FILE "testOutput"
#define INIT_LAMBDA 0.00001
#define LAMBDA_RANGE 10
#define GRADIENT_DESCENT_ITERATIONS 1000

void dataManager(int argc, char *argv[]);
void trainingDataToArr(char *num, double ***X, double **y, int *n, int *m);
void CVDataToArr(char *num, double ***X, double **y, int *n, int m);
void testDataToArr(char *num, double ***X, double **y, int *n, int m);
void initTheta(double *theta, const int m);
void gradientDescent(double **X, double *y, double *theta, double lambda, int n, const int m);
void hypothesis(double *h, double **X, double *theta, int n, const int m);
double partialDerivative(double *h, double **X, double *y, int n, int i);
double testAccuracy(double *h, double  *y, int n);
double mean(double *y, int n);
void freeVar(double ***X, double **y, int n);
int main(int argc, char *argv[]) {
    dataManager(argc, argv);
    double *allTheta[argc - 1];
    double allLambda[argc - 1];

    for (int i = 1; i < argc; ++i) {
        char num[2];
        num[0] = '0' + i;
        num[1] = 0;

        double lambda = INIT_LAMBDA;
        double accuracyResults[LAMBDA_RANGE];
        double **training_X, *training_y, **CV_X, *CV_y, **test_X, *test_y;
        int trainingSize, CVSize, testSize, m;
        trainingDataToArr(num, &training_X, &training_y, &trainingSize, &m);
        CVDataToArr(num, &CV_X, &CV_y, &CVSize, m);
        testDataToArr(num, &test_X, &test_y, &testSize, m);
        printf("Data has %d features. \n", m);
        double theta[m];
        double CV_h[CVSize];
        printf("Training set size = %d. \n", trainingSize);
        printf("Cross validation set size = %d. \n", CVSize);
        printf("Test set size = %d. \n", testSize);

        for (int j = 0; j < LAMBDA_RANGE; ++j) {
            initTheta(theta, m);
            gradientDescent(training_X, training_y, theta, lambda, trainingSize, m);
            hypothesis(CV_h, CV_X, theta, CVSize, m);
            accuracyResults[j] = testAccuracy(CV_h, CV_y, CVSize);
            lambda *= 10;
        }

        double max = accuracyResults[0], ind = 0;
        for (int j = 0; j < LAMBDA_RANGE; ++j) {
            if (accuracyResults[j] > max) {
                max = accuracyResults[j];
                ind = j;
            }
        }


        allLambda[i] = 0.0000001 * pow(10, ind);

        printf("The optimum value of lambda is %lf. \n", allLambda[i]);

        initTheta(theta, m);
        gradientDescent(test_X, test_y, theta, allLambda[i], testSize, m);
        allTheta[i] = theta;

        double test_h[testSize];
        hypothesis(test_h, test_X, theta, testSize, m);

        printf("Predicted Value           Actual Value\n");

        for (int j = 0; j < testSize; ++j) {
            printf("%lf %24lf\n", test_h[j], test_y[j]);
        }
        printf("The test accuracy is %lf. \n", testAccuracy(test_h, test_y, testSize));

        freeVar(&training_X, &training_y, trainingSize);
        freeVar(&CV_X, &CV_y, CVSize);
        freeVar(&test_X, &test_y, testSize);
    }
    return 0;
}

void dataManager(int argc, char *argv[]) {
    if (argc == 1) {
        puts("Not enough arguments. ");
        puts("Expected Format: exec_name.exe \"dataset1.csv\", \"dataset2.csv\" ... ");
        exit(EXIT_FAILURE);
    }
    char str[STR_BUFFER] = "C:\\Users\\agbod\\GitHub\\Machine_Learning\\Data\\";
    char buff[STR_BUFFER];
    strcpy(buff, str);
    strcat(str, "datamanager.py");
    for (int i = 1; i < argc; i++) {
        strcat(str, " ");
        strcat(str, buff);
        strcat(str, argv[i]);
    }
    system(str);
}

void trainingDataToArr(char *num, double ***X, double **y, int *n, int *m) {
    char trainingInputFile[STR_BUFFER] = "", trainingOutputFile[STR_BUFFER] = "", str[STR_BUFFER] = "";
    char *token;

    strcat(trainingInputFile, TRAINING_INPUT_FILE);
    strcat(trainingOutputFile, TRAINING_OUTPUT_FILE);
    strcat(trainingInputFile, num);
    strcat(trainingOutputFile, num);
    strcat(trainingInputFile, ".csv");
    strcat(trainingOutputFile, ".csv");

    *n = -1;
    *m = 0;
    FILE *fp = fopen(trainingInputFile, "r");
    fgets(str, STR_BUFFER, fp);
    token = strtok(str, " ");
    while (token != NULL) {
        (*m)++;
        token  = strtok(NULL, " ");
    }
    rewind(fp);
    while (!feof(fp)) {
        fgets(str, STR_BUFFER, fp);
        (*n)++;
    }

    *X = (double **)malloc(*n * sizeof(double *));
    for (int i = 0; i < *n; ++i) {
        (*X)[i] = (double *)malloc(*m * sizeof(double));
    }
    *y = (double *)malloc(*n * sizeof(double));
    rewind(fp);
    for (int i = 0; i < *n; ++i) {
        for (int j = 0; j < *m; ++j) {
            fscanf(fp, "%lf", &((*X)[i][j]));
        }
    }
    fclose(fp);
    fp = fopen(trainingOutputFile, "r");
    for (int i = 0; i < *n; ++i) {
        fscanf(fp, "%lf", &((*y)[i]));
    }
    fclose(fp);
}

void CVDataToArr(char *num, double ***X, double **y, int *n, int m) {
    char CVInputFile[STR_BUFFER] = "", CVOutputFile[STR_BUFFER] = "", str[STR_BUFFER] = "";

    strcat(CVInputFile, CV_INPUT_FILE);
    strcat(CVOutputFile, CV_OUTPUT_FILE);
    strcat(CVInputFile, num);
    strcat(CVOutputFile, num);
    strcat(CVInputFile, ".csv");
    strcat(CVOutputFile, ".csv");

    *n = -1;
    FILE *fp = fopen(CVInputFile, "r");
    fgets(str, STR_BUFFER, fp);
    while (!feof(fp)) {
        fgets(str, STR_BUFFER, fp);
        (*n)++;
    }

    *X = (double **)malloc(*n * sizeof(double *));
    for (int i = 0; i < *n; ++i) {
        (*X)[i] = (double *)malloc(m * sizeof(double));
    }
    *y = (double *)malloc(*n * sizeof(double));
    rewind(fp);
    for (int i = 0; i < *n; ++i) {
        for (int j = 0; j < m; ++j) {
            fscanf(fp, "%lf", &((*X)[i][j]));
        }
    }
    fclose(fp);
    fp = fopen(CVOutputFile, "r");
    for (int i = 0; i < *n; ++i) {
        fscanf(fp, "%lf", &((*y)[i]));
    }
    fclose(fp);
}

void testDataToArr(char *num, double ***X, double **y, int *n, int m) {
    char testInputFile[STR_BUFFER] = "", testOutputFile[STR_BUFFER] = "", str[STR_BUFFER] = "";

    strcat(testInputFile, TEST_INPUT_FILE);
    strcat(testOutputFile, TEST_OUTPUT_FILE);
    strcat(testInputFile, num);
    strcat(testOutputFile, num);
    strcat(testInputFile, ".csv");
    strcat(testOutputFile, ".csv");

    *n = -1;
    FILE *fp = fopen(testInputFile, "r");
    fgets(str, STR_BUFFER, fp);
    while (!feof(fp)) {
        fgets(str, STR_BUFFER, fp);
        (*n)++;
    }

    *X = (double **)malloc(*n * sizeof(double *));
    for (int i = 0; i < *n; ++i) {
        (*X)[i] = (double *)malloc(m * sizeof(double));
    }
    *y = (double *)malloc(*n * sizeof(double));
    rewind(fp);
    for (int i = 0; i < *n; ++i) {
        for (int j = 0; j < m; ++j) {
            fscanf(fp, "%lf", &((*X)[i][j]));
        }
    }
    fclose(fp);
    fp = fopen(testOutputFile, "r");
    for (int i = 0; i < *n; ++i) {
        fscanf(fp, "%lf", &((*y)[i]));
    }
    fclose(fp);
}

void initTheta(double *theta, const int m) {
    for (int i = 0; i < m; ++i) {
        theta[i] = 0;
    }
}

void gradientDescent(double **X, double *y, double *theta, double lambda, int n, const int m) {
    double h[n], temp[m];
    double learningRate = 0.00001;
    int iterator, convergence = 0;
    while (convergence == 0) {
        for (iterator = 0; iterator < GRADIENT_DESCENT_ITERATIONS; ++iterator) {
            hypothesis(h, X, theta, n, m);
            convergence = 1;
            for (int i = 0; i < m; ++i) {
                if (i == 0) {
                    temp[i] = theta[i] - (learningRate * partialDerivative(h, X, y, n, i));
                } else {
                    temp[i] = (theta[i] * (1 - (learningRate * lambda) / n)) - (learningRate * partialDerivative(h, X, y, n, i));
                }
                if (theta[i] - temp[i] > 0.00001 || theta[i] - temp[i] < -0.00001) {
                    convergence = 0;
                }
                theta[i] = temp[i];
            }
            if (convergence == 1) {
                break;
            }
        }
        learningRate *= 3;
        if (convergence == 1) {
            printf("For lambda %lf gradient descent converged at %d iterations with a learning rate %lf. \n", lambda, iterator, learningRate);
        }
    }
}

void hypothesis(double *h, double **X, double *theta, int n, const int m) {
    for (int i = 0; i < n; ++i) {
        double sum = 0;
        for (int j = 0; j < m; ++j) {
            sum += theta[j] * X[i][j];
        }
        h[i] = sum;
    }
}

double partialDerivative(double *h, double **X, double *y, int n, int i) {
    double sum = 0;
    for (int j = 0; j < n; ++j) {
        sum += (h[j] - y[j]) * X[j][i];
    }
    sum /= n;
    return sum;
}

double testAccuracy(double *h, double  *y, int n) {
    double avg = mean(y, n), sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; ++i) {
        sum1 += pow((y[i] - h[i]), 2);
    }
    for (int i = 0; i < n; ++i) {
        sum2 += pow((y[i] - avg), 2);
    }
    return 1 - (sum1 / sum2);
}

double mean(double *y, int n) {
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += y[i];
    }
    return sum / n;
}

void freeVar(double ***X, double **y, int n) {
    for (int i = 0; i < n; ++i) {
        free((*X)[i]);
    }
    free(*X);
    free(*y);
}
