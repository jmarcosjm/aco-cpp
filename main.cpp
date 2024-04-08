#include <iostream>
#include "NumCpp.hpp"
#include <sstream>
#include <ctime>
#include <random>

using namespace std;

//echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
//perf stat -d ./tcc1 heart_failure

// gprof ./tcc1 ../haberman.csv > profile.txt

static  map<std::string, clock_t> tempos;
static  vector<clock_t> tempos_get_probabilities_paths_ordered;
static  vector<clock_t> tempos_get_pheromone_deposit;




inline std::mt19937& generator() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}


template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
T my_rand(T min, T max) {
    std::uniform_int_distribution<T> dist(min, max);
    return dist(generator());
}


vector<vector<string>> from_nd_arrayD(nc::NdArray<double> ndArray){
    size_t numRows = ndArray.shape().rows;
    size_t numCols = ndArray.shape().cols;
    vector<vector<string>> dataframe;

    for (size_t i = 0; i < numRows; ++i) {
        vector<string> row;
        for (size_t j = 0; j < numCols; ++j) {
            row.push_back(to_string(ndArray(i, j)));
            cout << to_string(ndArray(i, j)) << "\t";
        }
        cout << endl;
        dataframe.push_back(row);
    }
    return dataframe;
}



vector<vector<string>> from_nd_array(nc::NdArray<int> ndArray){
    size_t numRows = ndArray.shape().rows;
    size_t numCols = ndArray.shape().cols;
    vector<vector<string>> dataframe;

    for (size_t i = 0; i < numRows; ++i) {
        vector<string> row;
        for (size_t j = 0; j < numCols; ++j) {
            row.push_back(to_string(ndArray(i, j)));
            cout << to_string(ndArray(i, j)) << "\t";
        }
        cout << endl;
        dataframe.push_back(row);
    }
    return dataframe;
}



vector<vector<string>> read_csv(const string filename, const char delimiter) {
    clock_t start = clock();
    vector<vector<string>> data;
    ifstream file(filename);
    if (file) {
        string line;
        while (getline(file, line)) {
            vector<string> row;
            stringstream ss(line);
            string cell;

            while (getline(ss, cell, delimiter)) {
                row.push_back(cell);
            }
            data.push_back(row);
        }
        file.close();
    }
    tempos["read_csv"] = clock () - start;
    return data;
}



size_t get_collumn_idx(const string& collumn_name, vector<vector<string>> dataframe){
    clock_t start = clock();
    for (size_t i = 0; i < dataframe[0].size(); i++) {
        if (dataframe[0][i] == collumn_name) {
            return i;
        }
    }
    tempos["get_collumn_idx"] = clock () - start;
    return 0;
}



void delete_collumn(const string& collumn_name, vector<vector<string>>& dataframe){
    clock_t start = clock();
    size_t collumn_idx = get_collumn_idx(collumn_name, dataframe);
    for (size_t i = 0; i < dataframe.size(); i++) {
        dataframe[i].erase(dataframe[i].begin() + collumn_idx);
    }
    tempos["delete_collumn"] = clock () - start;
}



void to_csv(const nc::NdArray<int>& instances){
    clock_t start = clock();
    size_t numRows = instances.shape().rows;
    size_t numCols = instances.shape().cols;
    ofstream outfile("../solutions.csv");


    for (size_t i = 0; i < numRows; ++i) {
        vector<string> row;
        for (size_t j = 0; j < numCols; ++j) {
            outfile << instances(i,j);
            if((j + 1) != numCols)
                outfile << ";";
        }
        outfile << endl;
    }
    tempos["to_csv"] = clock () - start;
}



nc::NdArray<double> to_nd_array(vector<vector<string>> dataframe){
    clock_t start = clock();
    size_t numRows = dataframe.size();
    size_t numCols = dataframe[0].size();
    nc::NdArray<double> data_array(numRows -1, numCols);

    for (size_t i = 1; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            data_array(i-1, j) = stod(dataframe[i][j]);
        }
    }
    tempos["to_nd_array"] = clock () - start;
    return data_array;
}



nc::NdArray<double> get_pairwise_distance(nc::NdArray<double> matrix) {
    clock_t start = clock();
    nc::int32 n = matrix.shape().rows;
    nc::NdArray<double> distances = nc::zeros<double>(n, n);
    for (nc::int32 i = 0; i < n; i++) {
        for (nc::int32 j = 0; j < n; j++) {
            double dot_product_x = 0;
            double dot_product_y = 0;
            double dot_product_xy = 0;
            for (nc::int32 k = 0; k < matrix.shape().cols; k++) {
                dot_product_x += std::pow(matrix(i, k), 2);
                dot_product_y += std::pow(matrix(j, k), 2);
                dot_product_xy += matrix(i, k) * matrix(j, k);
            }
            double distance = std::sqrt(dot_product_x - 2 * dot_product_xy + dot_product_y);
            distances(i, j) = distance;
            distances(j, i) = distance;
        }
    }
    tempos["get_pairwise_distance"] = clock () - start;
    return distances;
}



double get_pheromone_deposit(
        const std::vector<std::pair<int, int>>& ant_choices,
        const nc::NdArray<double>& distances,
        double deposit_factor
) {
    clock_t start = clock();

    double tour_length = 0;
    for (const auto& path : ant_choices) {
        tour_length += distances(path.first, path.second);
    }

    if (tour_length == 0) {
        return 0;
    }

    if (std::isinf(tour_length)) {
        std::cout << "deu muito ruim!" << std::endl;
    }

    tempos_get_pheromone_deposit.push_back(clock () - start);
    return deposit_factor / tour_length;
}



nc::NdArray<double> get_visibility_rates_by_distances(nc::NdArray<double> distances){
    clock_t start = clock();
    nc::NdArray<double> visibilities = nc::zeros<double>(distances.shape());

    for (nc::int32 i = 0; i < distances.shape().rows; ++i) {
        for (nc::int32 j = 0; j < distances.shape().cols; ++j) {
            if(i != j){
                if(distances(i,j) == 0) visibilities(i,j) = 0;
                else visibilities(i,j) = 1 / distances(i, j);
            }
        }
    }
    tempos["get_visibility_rates_by_distances"] = clock () - start;
    return visibilities;
}



nc::NdArray<int> create_colony(nc::uint32 num_ants) {
    clock_t start = clock();
   auto colonia = nc::full({num_ants, num_ants}, -1);
    tempos["create_colony"] = clock () - start;
   return colonia;
}



nc::NdArray<double> create_pheromone_trails(const nc::NdArray<double>& search_space, double initial_pheromone) {
    clock_t start = clock();
    auto trails = nc::full(search_space.shape(), initial_pheromone);
    nc::fillDiagonal(trails, 0.0);
    tempos["create_pheromone_trails"] = clock () - start;
    return trails;
}



std::vector<std::pair<int, double>> get_probabilities_paths_ordered(
        const nc::NdArray<int>& ant,
        const nc::NdArray<double>& visibility_rates,
        const nc::NdArray<double>& phe_trails) {
    clock_t start = clock();
    std::vector<int> available_instances;

    for (int i = 0; i < ant.size(); i++) {
        if (ant[i] < 0) {
            available_instances.push_back(i);
        }
    }

    double smell = 0.0;
    for (int i : available_instances) {
        smell += phe_trails[i] * visibility_rates[i];
    }

    std::vector<std::pair<int, double>> probabilities(available_instances.size());

    for (int i = 0; i < available_instances.size(); i++) {
        int available_instance = available_instances[i];
        probabilities[i].first = available_instance;
        double path_smell = phe_trails[available_instance] * visibility_rates[available_instance];
        if (path_smell == 0) {
            probabilities[i].second = 0;
        } else {
            probabilities[i].second = path_smell / smell;
        }
    }

    std::sort(probabilities.begin(), probabilities.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second > b.second;
              });


    tempos_get_probabilities_paths_ordered.push_back(clock () - start);
    return probabilities;
}



nc::NdArray<int> run_colony(nc::NdArray<double> X, int initial_pheromone, double evaporation_rate, int Q){
    clock_t start = clock();
    nc::NdArray<double> distances = get_pairwise_distance(X);
    nc::NdArray<double> visibility_rates = get_visibility_rates_by_distances(distances);
    nc::NdArray<int> the_colony = create_colony(X.shape().rows);
    for (nc::int32 i = 0; i < X.shape().rows; ++i) {
        the_colony(i,i) = 1;
    }
    std::vector<std::vector<std::pair<int, int>>> ant_choices;
    for (int i = 0; i < the_colony.shape().rows; ++i) {
        ant_choices.push_back({{i, i}});
    }
    nc::NdArray<double> pheromone_trails = create_pheromone_trails(distances, initial_pheromone);

    while (nc::any(the_colony == -1).item()) {

        for (int i = 0; i < the_colony.shape().rows; ++i) {
            auto ant = the_colony(i, the_colony.cSlice());
            if (nc::any(ant == -1).item()) {
                std::pair<int, int> last_choice = ant_choices[i].back();
                int ant_pos = last_choice.second;
                std::vector<std::pair<int, double>> choices = get_probabilities_paths_ordered(
                        ant,
                        visibility_rates(ant_pos, visibility_rates.cSlice()),
                        pheromone_trails(ant_pos, pheromone_trails.cSlice())
                );

                for (const auto& choice : choices) {
                    int next_instance = choice.first;
                    double probability = choice.second;
                    int ajk = my_rand(0, 1);
                    double final_probability = probability * ajk;
                    if (final_probability != 0) {
                        ant_choices[i].emplace_back(ant_pos, next_instance);
                        the_colony(i, next_instance) = 1;
                        break;
                    } else {
                        the_colony(i, next_instance) = 0;
                    }
                }
            }
        }
        for (nc::uint32 i = 0; i < the_colony.shape().rows; ++i) {
            double ant_deposit = get_pheromone_deposit(ant_choices[i], distances, Q);
            for (size_t j = 1; j < ant_choices[i].size(); ++j) { // Never deposit pheromone on i == j!
                auto path = ant_choices[i][j];
                pheromone_trails(path.first, path.second) += ant_deposit;
            }
        }

        // Pheromone evaporation
        for (nc::uint32 i = 0; i < pheromone_trails.shape().rows; ++i) {
            for (nc::uint32 j = 0; j < pheromone_trails.shape().cols; ++j) {
                pheromone_trails(i, j) = (1 - evaporation_rate) * pheromone_trails(i, j);
            }
        }

    }
    tempos["run_colony"] = clock() - start;
    return the_colony;
}

//perf stat -d ./tcc1 heart_failure

int main(int argc, char ** argv) {
    string classe = "DEATH_EVENT";
    const clock_t begin_time = clock();
    system("pwd");
    string path = "../datasets/" + (string) argv[argc -1] + ".csv";
    vector<vector<string>> dataframe = read_csv(path, ';');
    delete_collumn(classe, dataframe);
    int initial_pheromone = 1;
    int Q = 1;
    double evaporation_rate = 0.1;
    cout << "Starting Search" << endl;
    cout << path << endl;
    cout << dataframe.size() << endl;
    auto indices_selected = run_colony(to_nd_array(dataframe), initial_pheromone, evaporation_rate, Q);
    cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

    cout << "Execution finished" << endl;

    to_csv(indices_selected);

    for(const auto& pair : tempos) {
        std::cout << pair.first << ": " << pair.second/CLOCKS_PER_SEC << std::endl;
    }
    cout << "tempos_get_probabilities_paths_ordered: " << accumulate(tempos_get_probabilities_paths_ordered.begin(), tempos_get_probabilities_paths_ordered.end(), 0.0)/CLOCKS_PER_SEC << endl;
    cout << "tempos_get_pheromone_deposit: " << accumulate(tempos_get_pheromone_deposit.begin(), tempos_get_pheromone_deposit.end(), 0.0)/CLOCKS_PER_SEC << endl;
    return 0;
}
