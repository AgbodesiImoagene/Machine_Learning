#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define STR_BUFFER 2048
#define TRAINING_INPUT_FILE "trainingInput.csv"
#define TRAINING_OUTPUT_FILE "trainingOutput.csv"
#define CV_INPUT_FILE "CVInput.csv"
#define CV_OUTPUT_FILE "CVOutput.csv"
#define TEST_INPUT_FILE "testInput.csv"
#define TEST_OUTPUT_FILE "testOutput.csv"
#define INIT_LAMBDA 0.00001
#define LAMBDA_RANGE 10
#define INIT_ALPHA 0.00001
#define ALPHA_RANGE 150
#define GRADIENT_DESCENT_ITERATIONS 2000

void dataManager(int argc, char *argv[]);
void trainingDataToArr(double ***X, double **y, int *n, int *m);
void CVDataToArr(double ***X, double **y, int *n, int m);
void testDataToArr(double ***X, double **y, int *n, int m);
void initTheta(double *theta, int m);
void gradientDescent(double **X, double *y, double *theta, double lambda, int n, int m);
void hypothesis(double *h, double **X, double *theta, int n, int m);
double cost(double **X, double *y, double *theta, double lambda, int n, int m);
double partialDerivative(double *h, double **X, double *y, int n, int i);
double testAccuracy(double *h, double  *y, int n);
double precision(double *h, double  *y, int n);
double recall(double *h, double  *y, int n);
double F1score(double *h, double  *y, int n);
void freeVar(double ***X, double **y, int n);
int main(int argc, char *argv[]) {
    dataManager(argc, argv);

    double lambda = INIT_LAMBDA;
    double accuracyResults[LAMBDA_RANGE];
    double **training_X, *training_y, **CV_X, *CV_y, **test_X, *test_y;
    int trainingSize, CVSize, testSize, m;
    trainingDataToArr(&training_X, &training_y, &trainingSize, &m);
    CVDataToArr(&CV_X, &CV_y, &CVSize, m);
    testDataToArr(&test_X, &test_y, &testSize, m);
    printf("Data has %d features. \n", m);
    double theta[m];
    double CV_h[CVSize];
    printf("Training set size = %d. \n", trainingSize);
    printf("Cross validation set size = %d. \n", CVSize);
    printf("Test set size = %d. \n", testSize);

    for (int i = 0; i < LAMBDA_RANGE; ++i) {
        gradientDescent(training_X, training_y, theta, lambda, trainingSize, m);
        hypothesis(CV_h, CV_X, theta, CVSize, m);
        accuracyResults[i] = F1score(CV_h, CV_y, CVSize);
        lambda *= 10;
    }

    double max = accuracyResults[0], ind = 0;
    for (int i = 0; i < LAMBDA_RANGE; ++i) {
        if (accuracyResults[i] > max) {
            max = accuracyResults[i];
            ind = i;
        }
    }


    lambda = INIT_LAMBDA * pow(10, ind);

    printf("The optimum value of lambda is %lf. \n", lambda);

    initTheta(theta, m);
    gradientDescent(test_X, test_y, theta, lambda, testSize, m);

    double test_h[testSize];
    hypothesis(test_h, test_X, theta, testSize, m);
    printf("Theta\n");

    for (int i = 0; i < m; ++i) {
        printf("%lf\n", theta[i]);
    }

    printf("Predicted Value           Actual Value\n");

    for (int i = 0; i < testSize; ++i) {
        if (test_h[i] >= 0.5) {
            printf("1 %24lf\n", test_y[i]);
        } else {
            printf("0 %24lf\n", test_y[i]);
        }
    }
    printf("The test accuracy is %lf%% and the F1 Score is %lf. \n", testAccuracy(test_h, test_y, testSize), F1score(test_h, test_y, testSize));

    freeVar(&training_X, &training_y, trainingSize);
    freeVar(&CV_X, &CV_y, CVSize);
    freeVar(&test_X, &test_y, testSize);
    return 0;
}

void dataManager(int argc, char *argv[]) {
    char str[STR_BUFFER] = "C:\\Users\\agbod\\GitHub\\Machine_Learning\\Data\\";
    char buff[STR_BUFFER], file[STR_BUFFER];
    if (argc == 1) {
        printf("Enter the address of the dataset in the data folder: ");
        fgets(file, STR_BUFFER, stdin);
    } else {
        strcpy(file, argv[1]);
    }
    strcpy(buff, str);
    strcat(str, "DiscreteDataManager.py");
    strcat(str, " ");
    strcat(str, buff);
    strcat(str, file);
    system(str);
}

void trainingDataToArr(double ***X, double **y, int *n, int *m) {
    char trainingInputFile[] = TRAINING_INPUT_FILE;
    char trainingOutputFile[] = TRAINING_OUTPUT_FILE;
    char str[STR_BUFFER] = "";
    char *token;

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

void CVDataToArr(double ***X, double **y, int *n, int m) {
    char CVInputFile[] = CV_INPUT_FILE;
    char CVOutputFile[] = CV_OUTPUT_FILE;
    char str[STR_BUFFER] = "";

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

void testDataToArr(double ***X, double **y, int *n, int m) {
    char testInputFile[] = TEST_INPUT_FILE;
    char testOutputFile[] = TEST_OUTPUT_FILE;
    char str[STR_BUFFER] = "";

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
    double h[n], temp[m], bestTheta[ALPHA_RANGE][m];
    double learningRate = INIT_ALPHA;
    int iterator, convergence, divergence, j;
    for (j = 0; j < ALPHA_RANGE; ++j) {
        initTheta(theta, m);
        divergence = 0;
        for (iterator = 0; iterator < GRADIENT_DESCENT_ITERATIONS; ++iterator) {
            hypothesis(h, X, theta, n, m);
            convergence = 1;
            for (int i = 0; i < m; ++i) {
                if (i == 0) {
                    temp[i] = theta[i] - (learningRate * partialDerivative(h, X, y, n, i));
                } else {
                    temp[i] = (theta[i] * (1 - (learningRate * lambda) / n)) - (learningRate * partialDerivative(h, X, y, n, i));
                }
                if (fabs(theta[i] - temp[i]) > 0.000001) {
                    convergence = 0;
                }
            }
            if (iterator == 0) {
                if (cost(X, y, temp, lambda, n, m) > cost(X, y, theta, lambda, n, m)) {
                    divergence = 1;
                    break;
                }
            }
            for (int i = 0; i < m; ++i) {
                theta[i] = temp[i];
            }
            if (convergence == 1) {
                break;
            }
        }
        for (int i = 0; i < m; ++i) {
            bestTheta[j][i] = theta[i];
        }
        if (divergence == 1) {
            printf("For lambda %lf gradient descent diverged with a learning rate %lf. \n", lambda, learningRate);
            break;
        }
        if (convergence == 1) {
            printf("For lambda %lf gradient descent converged at %d iterations with a learning rate %lf. \n", lambda, iterator, learningRate);
            break;
        }
        learningRate *= 3;
    }
    if (convergence != 1) {
        double accuracyResults[j];
        for (int i = 0; i < j; ++i) {
            hypothesis(h, X, bestTheta[i], n, m);
            accuracyResults[i] = F1score(h, y, n);
        }
        double max = accuracyResults[0];
        int maxInd = 0;
        for (int i = 0; i < j; ++i) {
            if (accuracyResults[i] > max) {
                max = accuracyResults[i];
                maxInd = i;
            }
        }
        for (int i = 0; i < m; ++i) {
            theta[i] = bestTheta[maxInd][i];
        }
    }
}

void hypothesis(double *h, double **X, double *theta, int n, const int m) {
    for (int i = 0; i < n; ++i) {
        double sum = 0;
        for (int j = 0; j < m; ++j) {
            sum += theta[j] * X[i][j];
        }
        h[i] = 1 / (1 + exp(-sum));
    }
}

double cost(double **X, double *y, double *theta, double lambda, int n, int m) {
    double h[n], sum = 0, reg = 0;
    hypothesis(h, X, theta, n, m);
    for (int i = 0; i < n; ++i) {
        sum += (y[i] * log(h[i])) + ((1 - y[i]) * log(1 - h[i]));
    }
    for (int i = 0; i < m; ++i) {
        reg += pow(theta[i], 2);
    }
    return ((sum * -1) / n) + ((reg * lambda) / (2 * n));
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
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        if (h[i] >= 0.5) {
            if (y[i] == 1) {
                sum++;
            }
        } else {
            if (y[i] == 0) {
                sum++;
            }
        }
    }
    return 100 * sum / n;
}

double precision(double *h, double  *y, int n) {
    double tp = 0, fp = 0;
    for (int i = 0; i < n; ++i) {
        if (h[i] >= 0.5 && y[i] == 1) {
            tp++;
        }
        if (h[i] >= 0.5 && y[i] == 0) {
            fp++;
        }
    }
    return  tp / (tp + fp);
}

double recall(double *h, double  *y, int n) {
    double tp = 0, fn = 0;
    for (int i = 0; i < n; ++i) {
        if (h[i] >= 0.5 && y[i] == 1) {
            tp++;
        }
        if (h[i] < 0.5 && y[i] == 1) {
            fn++;
        }
    }
    return  tp / (tp + fn);
}

double F1score(double *h, double  *y, int n) {
    double prec = precision(h, y, n), rec = recall(h, y, n);
    return (2 * prec * rec) / (prec + rec);
}

void freeVar(double ***X, double **y, int n) {
    for (int i = 0; i < n; ++i) {
        free((*X)[i]);
    }
    free(*X);
    free(*y);
}
