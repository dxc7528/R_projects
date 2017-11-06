library(tidyverse)
library(corrr)
library(igraph)
library(ggraph)

tidy_cors <- mtcars %>% 
  correlate() %>% 
  stretch()


graph_cors <- tidy_cors %>%
  filter(abs(r) > .3) %>%
  graph_from_data_frame(directed = FALSE)


ggraph(graph_cors) +
  geom_edge_link() +
  geom_node_point() +
  geom_node_text(aes(label = name))


ggraph(graph_cors) +
  geom_edge_link(aes(edge_alpha = abs(r), edge_width = abs(r), color = r)) +
  guides(edge_alpha = "none", edge_width = "none") +
  scale_edge_colour_gradientn(limits = c(-1, 1), colors = c( "dodgerblue2","firebrick2")) +
  geom_node_point(color = "white", size = 7) +
  geom_node_text(aes(label = name), repel = TRUE) +
  theme_graph() +
  labs(title = "•]‰¿€–Ú‚Ì‘ŠŠÖ")
