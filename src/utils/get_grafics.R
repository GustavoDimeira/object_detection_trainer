library(multcompView)
library(data.table)
library("Metrics")
library("ggplot2")
library("gridExtra")

# variaveis globais

saving_path <- "./dataset"

results <- data.table(read.table("./results.csv", sep = ",", header = TRUE))
counting <- data.table(read.table("./counting.csv", sep = ",", header = TRUE))

metrics <- list("mAP50", "mAP75", "mAP", "precision", "recall", "fscore")
models <- c(unique(results$ml))

# +------------------------------------------+
# | Grafico relacao GroundThruth X Predicted |
# +------------------------------------------+

grafics <- list()

for (model in models) {
  ml_counting <- counting[counting$ml == model, ]
  max <- max(ml_counting$groundtruth, ml_counting$predicted)

  new_graf <- ggplot(ml_counting, aes(
    x = ml_counting$groundtruth, y = ml_counting$predicted
  )) +
    geom_point() + geom_smooth(method = "lm", formula = "y ~ x") +
    labs(title = model, x = "Real", y = "Previsto") +
    theme(plot.title = element_text(size = 10)) +
    xlim(0, max) + ylim(0, max)

  grafics[[length(grafics) + 1]] <- new_graf
}

plot <- grid.arrange(grobs = grafics, ncol = 2)

save <- file.path(saving_path, "counting.png")
ggsave(filename = save, plot, width = 10, height = 14)

# +--------------------------------------------+
# | Grafico distribuicao de objetos por imagem |
# +--------------------------------------------+

ml_counting <- counting[counting$ml == models[[1]], ]

new_graf <- ggplot(ml_counting, aes(x = groundtruth)) +
  geom_histogram(color = "darkblue", fill = "lightblue") +
  labs(title = "Historigram", x = "Contagem", y = "Densidate")

save <- file.path(saving_path, "historogram.png")
ggsave(filename = save, new_graf)

# +-----------------------------------------------------+
# | Gera arquivo anova.txt com valores do teste tukey e |
# | cld (Compact Letter Display).                       |
# +-----------------------------------------------------+

clds <- list()

sink(file.path(saving_path, "anova.txt"))

for (metric in metrics) {
  formula <- as.formula(paste0("results$", metric, " ~ results$ml"))
  tukey_result <- TukeyHSD(aov(formula), conf.level = 0.95)

  cld <- multcompLetters4(aov(formula), tukey_result)
  clds[[metric]] <- cld

  cat("+-------------------[ Teste para", metric, "]-------------------+\n")
  print(tukey_result)
  print(cld)
}

sink()

# +----------------------------------------------------------------------+
# | Gera arquivo statistics.txt com valores de media, mediana, iqr e std |
# +----------------------------------------------------------------------+

sink(file.path(saving_path, "/statistics.txt"))

for (metric in metrics) {
  statistcs <- results[, .(
    median = median(get(metric), na.rm = TRUE),
    IQR = IQR(get(metric), na.rm = TRUE),
    mean = mean(get(metric), na.rm = TRUE),
    sd = sd(get(metric), na.rm = TRUE)
  ), by = ml]

  cat("\n+----------[ Estatísticas para", metric, "]----------+\n")
  print(statistcs)
}

sink()

# +-----------------------------------------+
# | Grafico Box-Plot do arquivo results.csv |
# +-----------------------------------------+

grafics <- c()

for (metric in metrics) {
  letters <- clds[[metric]]$"results$ml"$Letters
  print(letters)

  g <- ggplot(results, aes(x = ml, y = get(metric), fill = ml)) +
    geom_boxplot() +
    stat_summary(
      fun = max,
      geom = "text", aes(label = letters[ml]),
      color = "black", size = 6,
    ) +
    scale_fill_brewer(palette = "Set1") +
    labs(title = metric, x = "", y = "") +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 14),
      axis.text.y = element_text(angle = 0, hjust = 0.5, size = 14),
      plot.title = element_text(size = 16)
    )

  grafics[[length(grafics) + 1]] <- g

  save <- file.path(saving_path, "grafics", paste0(metric, ".png"))
  ggsave(filename = save, g, width = 14, height = 8)
}

g <- grid.arrange(grobs = grafics, ncol = 3)

save <- file.path(saving_path, "boxplot.png")
ggsave(filename = save, g, width = 14, height = 8)
