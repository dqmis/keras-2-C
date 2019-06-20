#include <stdio.h>
#include <stdlib.h>

#define layers_file "layers.txt"
#define scale_file "scaler.txt"
#define input_file "inputs.txt"
#define result_file "predictions.txt"

const int input_dim = 1;
const int input_count = 5;
const int layer_count = 3;

// layer structure
typedef struct
{
    int layer_count; // count of layer
    int input; // number of last layer's nodes / input vars count
    int unit; // number of next layer's nodes count
    float *b; // bias of layer
    float **w; // weights of layer
} layer;

// allocates memory for 2D array (matrix)
float **get_arr(int input_dim, int unit)
{
    float **output = (float **)malloc(input_dim * sizeof(float *));
    for (int j = 0; j < input_dim; j++)
        output[j] = malloc(unit * sizeof * output[j]);
    return output;
}

// layer constructor
void init_layer_types(layer* l, int layer_count, int input, int unit)
{
    l->input = input;
    l->unit = unit;
    l->layer_count = layer_count;

    l->w = get_arr(input, unit);
    l->b = (float *)malloc(unit * sizeof(float));
}

// reads layers' information from layers_file
void read_layers(int layers_count, layer network[])
{
    FILE* file = fopen(layers_file, "r");
    int input, unit;
    float val;
    for (int i = 0; i < layers_count; i++)
    {
        layer l;
        fscanf(file, "%d", &input);
        fscanf(file, "%d", &unit);
        init_layer_types(&l, i, input, unit);

        for (int j = 0; j < input; j++)
            for (int k = 0; k < unit; k++)
            {
                fscanf(file, "%f", &val);
                l.w[j][k] = val;
            }
        for (int j = 0; j < unit; j++)
        {
            fscanf(file, "%f", &val);
            l.b[j] = val;
        }
        network[i] = l;
    }
    fclose(file);
}

// multiplicates two matrixes into one
// m(n) - rows count
// n(n) - columns count
void dot(int m1, int m2, int n1, int n2, float **mat1, float **mat2, float **res)
{
    for (int i = 0; i < m1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            res[i][j] = 0;
            for (int x = 0; x < m2; x++)
                res[i][j] += mat1[i][x] * mat2[x][j];
        }
    }
}

// adds vector to matrix
// n(n) - rows count of matrix (must be same as size of b vector)
void add(float **output, float *b, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            output[i][j] += b[j];
}

// reads input variables to input matrix
// dim - count of inputs per row
void read_inputs(int dim, float **inputs)
{
    float val;
    FILE* file = fopen(input_file, "r");
    for (int i = 0; i < dim; i++)
    {
        fscanf(file, "%f", &val);
        inputs[0][i] = val;
    }
    fclose(file);
}

// scales inputs based on sklearn StandardScaler transform method
// reads scaling vars from scale_file
void scale_inputs(int dim, float **inputs)
{
    float val;
    float s;
    FILE* file = fopen(scale_file, "r");
    for (int i = 0; i < dim; i++)
    {
        fscanf(file, "%f", &val);
        inputs[0][i] -= val;
    }
    for (int i = 0; i < dim; i++)
    {
        fscanf(file, "%f", &val);
        inputs[0][i] /= val;
    }
    fclose(file);
}

// frees memory of 2D matrix and sets pointer to NULL
void free_memory(int n, float **arr)
{
    for (int i = 0; i < n; i++)
        free(arr[i]);
    free(arr);
    arr = NULL;
}

// Relu activation
void relu(int input_dim, int unit, float** input)
{
    for (int i = 0; i < input_dim; i++)
        for (int j = 0; j < unit; j++)
            if (input[i][j] <= 0) input[i][j] = 0;
}

// writes predictions to result_file
void write_res(float res)
{
    FILE* file = fopen(result_file, "a");
    fprintf(file, "%f\n", res);
    fclose(file);
}

int main() {
    float val; // var for storing fscan values
    int count; // number of input rows

    // reads information of neural new layers    
    layer *network = (layer *)malloc(layer_count * sizeof(layer));
    read_layers(layer_count, network);

    // scans the number of input rows
    FILE* file = fopen(input_file, "r");
    fscanf(file, "%d", &count);

    // in the loop matrix operations are made for each input row
    for (int i = 0; i < count; i++) {
        float **inputs = get_arr(input_dim, input_count);

        // reads row to input array
        for (int i = 0; i < input_count; i++) {
            fscanf(file, "%f", &val);
            inputs[0][i] = val;
        }

        // scaling inputs based of scale file
        scale_inputs(input_count, inputs);

        float  **output;
        for (int i = 0; i < layer_count; i++)
        {
            output = get_arr(input_dim, network[i].unit);

            // making layer step calulations and activation
            dot(input_dim, network[i].input, network[i].input, network[i].unit, inputs, network[i].w, output);
            add(output, network[i].b, input_dim, network[i].unit);
            relu(input_dim, network[i].unit, output);

            // releasing memory for input array
            free_memory(input_dim, inputs);
            inputs = output;
        }

        // writing result to file
        write_res(*(inputs[0]));
        free_memory(input_dim, inputs);
    }
    fclose(file);
    return 0;
}
