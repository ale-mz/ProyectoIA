# Copyright Oscar Fernández, Jorge Madríz y Kenneth Villalobos

install.packages("lubridate")
install.packages("dplyr")
install.packages("tidyr")
install.packages("ggplot2")
install.packages("reshape2")
install.packages("scales")
library(lubridate)
library(dplyr)
library(tidyr)
library(ggplot2)
library(reshape2)
library(scales)
# ------------------------------- Data reading --------------------------------

# Read the dataset
news <- (read.csv("../../datasets/Dataset_noticias.csv", header=T, encoding = "UTF-8"))


# ------------------------------- Data formatting -----------------------------

# Delete the unnecessary columns 
news_cleaned <- news %>%
  select(
    -Revisada.por,
    -Fuente,
    -Autor
  )

# Formta the date, add a column to the number of characters per news
news_cleaned <- news_cleaned %>%
  mutate(Fecha.publicacion = dmy(Fecha.publicacion),
          Longitud.noticia = nchar(Noticia))

# ------------------------------- Table construction --------------------------

# Table for the number of news per cause
cause_counts <- news_cleaned %>%
  summarise(
    Humano = sum(Humano),
    Vía.y.entorno = sum(Vía.y.entorno),
    Vehicular = sum(Vehicular)
  ) %>%
  gather(key = "Causa", value = "Noticias") %>%
  arrange(desc(Noticias))

# Show the table
cause_counts



# Table for the proportion of news from each cause interaction
cause_proportions <- news_cleaned %>%
  summarise(
    Humano.Via.Vehicular = sum(Humano & Vía.y.entorno & Vehicular),
    
    Humano.Via = sum(Humano & Vía.y.entorno) -
      Humano.Via.Vehicular,
    Humano.Vehicular = sum(Humano & Vehicular) -
      Humano.Via.Vehicular,
    Via.Vehicular = sum(Vía.y.entorno & Vehicular) -
      Humano.Via.Vehicular,
    
    Humano = sum(Humano) -
      Humano.Via.Vehicular -
      Humano.Via -
      Humano.Vehicular,
    Vía.y.entorno = sum(Vía.y.entorno) -
      Humano.Via.Vehicular -
      Humano.Via -
      Via.Vehicular,
    Vehicular = sum(Vehicular) -
      Humano.Via.Vehicular -
      Humano.Vehicular -
      Via.Vehicular
  ) %>%
  gather(key = "Causas", value = "Noticias") %>%
  mutate(Proporcion = (Noticias / sum(Noticias)) * 100,
         Causas = factor(Causas, levels = unique(Causas)))

# Show the table
cause_proportions



# Table for the number and proportions of news from
# known causes vs unknown causes
known_vs_unknown_proportions <- news_cleaned %>%
  summarise(
    Conocida = sum(Humano | Vía.y.entorno | Vehicular),
    Desconocida = n() - Conocida
  ) %>%
  gather(key = "Causa", value = "Noticias") %>%
  arrange(desc(Noticias)) %>%
  mutate(Proporcion = (Noticias / sum(Noticias)) * 100,
         Causa = factor(Causa, levels = unique(Causa)))

# Show the table
known_vs_unknown_proportions



# Table for the number and proportions of news from each year
year_proportions <- news_cleaned %>%
  count(Año.publicacion = year(Fecha.publicacion), name = "Noticias") %>%
  arrange(desc(Noticias)) %>%
  mutate(Proporcion = Noticias / sum(Noticias) * 100)

# Show the table
year_proportions



# Table for the number and proportions of news from each reporter
reporter_proportions <- news_cleaned %>%
  count(Noticiero, name = "Noticias") %>%
  arrange(desc(Noticias)) %>%
  mutate(Proporcion = Noticias / sum(Noticias) * 100)

# Show the table
reporter_proportions



# Table with the correlation matrix
correlation_matrix <- cor(news_cleaned %>%
                            select(Humano, Vía.y.entorno, Vehicular))

# Show the table
correlation_matrix

# ------------------------------- Graph construction --------------------------

# ******************************* Bar graphs **********************************

# Bar graph with the number of news per cause
ggplot(cause_counts, aes(x = reorder(Causa, Noticias), y = Noticias)) +
  geom_bar(stat = "identity", fill = "tomato") +
  labs(x = "", y = "Cantidad de noticias") +
  theme_minimal() +
  coord_flip() +
  theme(
    axis.title.x = element_text(margin = margin(t = 12, r = 0, b = 0, l = 0))
  )



# Bar graph with the number of news with known causes vs unknown causes
ggplot(known_vs_unknown_proportions, aes(x = reorder(Causa, Noticias), y = Noticias)) +
  geom_bar(stat = "identity", fill = "sandybrown") +
  labs(x = "", y = "Cantidad de noticias") +
  theme_minimal() +
  coord_flip() +
  theme(
    axis.title.x = element_text(margin = margin(t = 12, r = 0, b = 0, l = 0))
  )



# Bar graph with the number of news per year
ggplot(year_proportions, aes(x = reorder(Año.publicacion, Noticias), y = Noticias)) +
  geom_bar(stat = "identity", fill = "goldenrod1") +
  labs(x = "", y = "Cantidad de noticias") +
  theme_minimal() +
  coord_flip() +
  theme(
    axis.title.x = element_text(margin = margin(t = 12, r = 0, b = 0, l = 0))
  )



# Bar graph with the number of news per reporter
ggplot(reporter_proportions, aes(x = reorder(Noticiero, Noticias), y = Noticias)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(x = "", y = "Cantidad de noticias") +
  theme_minimal() +
  coord_flip() +
  theme(
    axis.title.x = element_text(margin = margin(t = 12, r = 0, b = 0, l = 0))
  )

# ******************************* Pie graphs **********************************

# Pie graph with the proportions of the news based on the causes
ggplot(cause_proportions, aes(x = "", y = Proporcion, fill = Causas)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0) +
  labs(fill = "Causas", y = "") +
  theme_minimal() +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.title.y = element_blank()
  ) +
  scale_fill_manual(
    values = c("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
               "#ffd92f", "#e5c494"),
    labels = paste0(cause_proportions$Causas,
                    " (", round(cause_proportions$Proporcion, 0), "%)"))

# ******************************* Density graphs ******************************

# Rearrange the information to create the graphs
news_density <- news_cleaned %>%
  gather(key = "Causa", value = "Presencia", Humano, Vía.y.entorno, Vehicular) %>%
  filter(Presencia == 1) %>%
  select(-Presencia)


# Density graph with the density of each cause over the years
ggplot(news_density, aes(x = Fecha.publicacion, fill = Causa)) +
  geom_density(alpha = 0.5) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  scale_y_continuous(labels = scales::comma) +
  labs(x = "Año de publicación",
       y = "Densidad",
       fill = "Causa") +
  theme_minimal() +
  theme(
    axis.title.x = element_text(margin = margin(t = 12, r = 0, b = 0, l = 0)),
    axis.title.y = element_text(margin = margin(t = 0, r = 12, b = 0, l = 0))
  )

# ******************************* Heatmap graphs ******************************

# Heatmap graph with the correlation between causes
ggplot(data = melt(correlation_matrix), aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "salmon", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Correlación") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed() +
  labs(x = "",
       y = "")

# ******************************* line graphs *********************************

# Rearrange the information to create the graphs
general_news_trend <- news_cleaned %>%
  mutate(Month = month(Fecha.publicacion, label=TRUE, abbr=TRUE)) %>%
  count(Month)

cause_news_trend <- news_cleaned %>%
  mutate(Month = month(Fecha.publicacion, label = TRUE, abbr = TRUE)) %>%
  gather(key = "Causa", value = "Presencia", Humano, Vía.y.entorno, Vehicular) %>%
  filter(Presencia == 1) %>%
  count(Month, Causa)


# Line graph with the tendency on the number of news over the year
ggplot(general_news_trend, aes(x = Month, y = n, group = 1)) +
  geom_line(color = "cyan4") +
  geom_point(color = "darkcyan") +
  labs(x = "",
       y = "Cantidad de noticias") +
  theme_minimal() +
  theme(
    axis.title.y = element_text(margin = margin(t = 0, r = 12, b = 0, l = 0))
  )

# Line graph with the tendency on the number of news over the year per cause
ggplot(cause_news_trend, aes(x = Month, y = n, color = Causa, group = Causa)) +
  geom_line() +
  geom_point() +
  labs(x = "",
       y = "Número de noticias",
       color = "Causa") +
  theme_minimal() +
  theme(
    axis.title.y = element_text(margin = margin(t = 0, r = 12, b = 0, l = 0))
  )

# ******************************* boxplot graphs ******************************

# Rearrange the information to create the graphs
news_boxplot <- news_cleaned %>%
  gather(key = "Causa", value = "Presencia", Humano, Vía.y.entorno, Vehicular) %>%
  filter(Presencia == 1) %>%
  select(Causa, Longitud.noticia)


# Boxplot graph with the number of characters per reporter
ggplot(news_cleaned, aes(x = Noticiero, y = Longitud.noticia, fill = Noticiero)) +
  geom_boxplot() +
  labs(x = "",
       y = "Cantidad de caracteres") +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.y = element_text(margin = margin(t = 0, r = 12, b = 0, l = 0))
  )

# Boxplot graph with the number of characters per cause
ggplot(news_boxplot, aes(x = Causa, y = Longitud.noticia, fill = Causa)) +
  geom_boxplot() +
  labs(x = "",
       y = "Cantidad de caracteres") +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.title.y = element_text(margin = margin(t = 0, r = 12, b = 0, l = 0))
  )

# -----------------------------------------------------------------------------