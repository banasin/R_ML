# Библиотеки
library(caret)
library(randomForest)
library(e1071)

# Загрузка данных
data <- read_delim("Wilms tumor.txt", delim = "\t", escape_double = FALSE, trim_ws = TRUE)

# Предполагаем, что первый столбец - это метки классов (рецидив: 1, нет рецидива: 0)
labels <- as.factor(ifelse(data[,1] == "Relapse", 1, 0))
features <- data[, -1]

# Отбор признаков с использованием теста t-Стьюдента
p_values <- apply(features, 2, function(x) t.test(x ~ labels)$p.value)
selected_features_ttest <- features[, p_values < 0.05]

# Рекурсивная элиминация признаков
set.seed(123)
ctrl <- rfeControl(functions=rfFuncs, method="cv", number=10)
rfe_result <- rfe(selected_features_ttest, labels, sizes=c(1:10), rfeControl=ctrl)

# Извлечение оптимального набора признаков
optimal_features <- selected_features_ttest[, rfe_result$optVariables]

# Разделение данных на обучающую и тестовую выборки
set.seed(123)
trainIndex <- createDataPartition(labels, p = .8, list = FALSE, times = 1)
train_data <- optimal_features[trainIndex, ]
test_data <- optimal_features[-trainIndex, ]
train_labels <- labels[trainIndex]
test_labels <- labels[-trainIndex]

# Моделирование
# K ближайших соседей
set.seed(123)
knn_model <- knn3(train_data, train_labels, k = 5)
knn_pred <- predict(knn_model, test_data)
knn_accuracy <- sum(knn_pred == test_labels) / length(test_labels)

# Метод опорных векторов
# Линейное ядро
set.seed(123)
svm_linear <- svm(train_data, train_labels, kernel = "linear")
svm_linear_pred <- predict(svm_linear, test_data)
svm_linear_accuracy <- sum(svm_linear_pred == test_labels) / length(test_labels)

# Радиальное ядро
set.seed(123)
svm_radial <- svm(train_data, train_labels, kernel = "radial")
svm_radial_pred <- predict(svm_radial, test_data)
svm_radial_accuracy <- sum(svm_radial_pred == test_labels) / length(test_labels)

# Random Forest
set.seed(123)
rf_model <- randomForest(train_data, train_labels)
rf_pred <- predict(rf_model, test_data)
rf_accuracy <- sum(rf_pred == test_labels) / length(test_labels)

# Вывод результатов
print(list(
  knn_accuracy = knn_accuracy,
  svm_linear_accuracy = svm_linear_accuracy,
  svm_radial_accuracy = svm_radial_accuracy,
  rf_accuracy = rf_accuracy
))