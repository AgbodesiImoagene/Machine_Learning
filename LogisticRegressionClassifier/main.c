#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STR_BUFFER 2048
#define TRAINING_INPUT_FILE "../data/trainingInput"
#define TRAINING_OUTPUT_FILE "../data/trainingOutput"
#define TEST_INPUT_FILE "../data/testInput"
#define TEST_OUTPUT_FILE "../data/testOutput"
#define GRADIENT_DESCENT_ITERATIONS 500

void dataManager(int argc, char *argv[]);
void gradientDescent(double **X, double *y, double *theta, int n, int m);
void hypothesis(double *h, double **X, double *theta, int n, int m);
double partialDerivative(double *h, double **X, double *y, int n, int i);
double testAccuracy(double *h, double **X, double  *y, int n);
int main(int argc, char *argv[]) {
    dataManager(argc, argv);
    double *allTheta[argc - 1];
    double allLambda[argc - 1];

    for (int i = 1; i < argc; ++i) {
        int n = -1, m = 0;
        char num[2];
        num[0] = '0' + i;
        num[1] = 0;

        char trainingInputFile[STR_BUFFER] = "", trainingOutputFile[STR_BUFFER] = "", str[STR_BUFFER] = "";
        char *token;

        strcat(trainingInputFile, TRAINING_INPUT_FILE);
        strcat(trainingOutputFile, TRAINING_OUTPUT_FILE);
        strcat(trainingInputFile, num);
        strcat(trainingOutputFile, num);
        strcat(trainingInputFile, ".csv");
        strcat(trainingOutputFile, ".csv");

        FILE *fp = fopen(trainingInputFile, "r");
        fgets(str, STR_BUFFER, fp);
        token = strtok(str, " ");
        while (token != NULL) {
            m++;
            token  = strtok(NULL, " ");
        }
        rewind(fp);
        while (!feof(fp)) {
            fgets(str, STR_BUFFER, fp);
            n++;
        }
        printf("%d\n", m);
        printf("%d\n", n);

        double **X = (double **)malloc(n * sizeof(double *));
        for (int j = 0; j < n; ++j) {
            X[j] = (double *)malloc(m * sizeof(double));
        }
        double *y = (double *)malloc(n * (sizeof(double)));
        rewind(fp);
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < m; ++k) {
                fscanf(fp, "%lf", &X[j][k]);
            }
        }
        fclose(fp);
        fp = fopen(trainingOutputFile, "r");
        for (int j = 0; j < n; ++j) {
            fscanf(fp, "%lf", &y[j]);
        }
        fclose(fp);
        puts("");


        for (int j = 0; j < 20; ++j) {
            printf("%lf", X[j][0]);
        }
        double theta[m];
        for (int j = 0; j < m; ++j) {
            theta[j] = 0;
        }

        gradientDescent(X, y, theta, n, m);
        free(*X);
        free(X);
        free(y);

        allTheta[i] = theta;
        puts("");
        for (int j = 0; j < m; ++j) {
            printf("%lf\n", theta[j]);
        }

        char testInputFile[STR_BUFFER] = "", testOutputFile[STR_BUFFER] = "";

        strcat(testInputFile, TEST_INPUT_FILE);
        strcat(testOutputFile, TEST_OUTPUT_FILE);
        strcat(testInputFile, num);
        strcat(testOutputFile, num);
        strcat(testInputFile, ".csv");
        strcat(testOutputFile, ".csv");
        puts(testOutputFile);

        fp = fopen(testInputFile, "r");
        n = -1;
        while (!feof(fp)) {
            fgets(str, STR_BUFFER, fp);
            n++;
        }
        printf("%d\n", n);
        rewind(fp);
        X = (double **)malloc(n * sizeof(double *));
        for (int j = 0; j < n; ++j) {
            X[j] = (double *)malloc(m * sizeof(double));
        }
        y = (double *)malloc(n * sizeof(double));

        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < m; ++k) {
                fscanf(fp, "%lf", &X[j][k]);
            }
        }
        fclose(fp);
        fp = fopen(testOutputFile, "r");
        for (int j = 0; j < n; ++j) {
            fscanf(fp, "%lf", &y[j]);
        }
        fclose(fp);
        double newH[n];
        hypothesis(newH, X, theta, n, m);
        printf("The test accuracy is %lf", testAccuracy(newH, X, y, n));
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
    puts(str);
}

void gradientDescent(double **X, double *y, double *theta, int n, int m) {
    double h[n], temp[m];
    double learningRate = 0.00001;
    int iterator, convergence = 0;
    while (convergence == 0) {
        for (iterator = 0; iterator < GRADIENT_DESCENT_ITERATIONS; ++iterator) {
            hypothesis(h, X, theta, n, m);
            convergence = 1;
            for (int i = 0; i < m; ++i) {
                temp[i] = theta[i] - (learningRate * partialDerivative(h, X, y, n, i));
                if (theta[i] - temp[i] > 0.001 || theta[i] - temp[i] < -0.001) {
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
            printf("Gradient descent converged at %d iterations. \n", iterator);
        }
    }
}

void hypothesis(double *h, double **X, double *theta, int n, int m) {
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

double testAccuracy(double *h, double **X, double  *y, int n) {+
            printf("Predicted Value           Actual Value\n");
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        printf("%lf %24lf\n", h[i], y[i]);
        sum += (h[i] - y[i]) * (h[i] - y[i]);
    }
    return sum / n;
}
