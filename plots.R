library(tidyverse)
dtf <- read.csv('results/test_results.csv') %>%
  mutate(k = factor(k))

# Dtv histogram
dtf %>% 
  ggplot(aes(x = dtv)) +
  geom_histogram(fill = 'navy', alpha = 0.7) +
  labs(title = "Histogram of dtv column", x ="", y = "") +
  theme_bw() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

# Dtv vs k grouped by alpha known / unknown
# Summary for mean dtv for different k
# dtf <- dtf %>% filter(Theta_ID == 2)
summary_k <- dtf %>% group_by(k) %>% summarise(mean_dtv = mean(dtv))

dtf_est_TRUE <- dtf %>% filter(est_alpha == 'True')
dtf_est_FALSE <- dtf %>% filter(est_alpha == 'False')

summary_k_eT <- dtf_est_TRUE %>% group_by(k) %>% summarise(mean_dtv = mean(dtv))
summary_k_eF <- dtf_est_FALSE %>% group_by(k) %>% summarise(mean_dtv = mean(dtv))

dtf %>% 
  ggplot(aes(y = dtv, x = k)) +
  geom_jitter(aes(col = est_alpha), size = 3, alpha = 0.5) +
  geom_line(data = summary_k, aes(x = k, y = mean_dtv, group = 1)) +
  geom_line(data = summary_k_eT, aes(x = k, y = mean_dtv, group = 1), col = 'navy') +
  geom_line(data = summary_k_eF, aes(x = k, y = mean_dtv, group = 1), col = 'red') +
  labs(title = "Dtv vs number of observations: Thetas set ID = 2", 
       x ="number of observations", y = "", subtitle = 'Grouped by est_alpha') +
  scale_color_manual(values=c("red", "navy")) +
  theme_bw() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

# Difference alpha_est
dtf_est_TRUE %>%
  ggplot(aes(x = alpha_dif, y = dtv)) +
  geom_jitter(alpha = 0.7, aes(col = init), size = 3) +
  facet_wrap(~Thetas_ID) +
  theme_bw() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
  labs(title = "Dtv vs difference between real and estimated alpha", x ="", y = "", subtitle = 'Init methods and distributions used to generate the data compared')

# Different init methods
dtf_random <- dtf %>% filter(init == 'random')
dtf_unif <- dtf %>% filter(init == 'uniform')
dtf_mean <- dtf %>% filter(init == 'mean')

summary_k_mean <- dtf_mean %>% group_by(k) %>% summarise(mean_dtv = mean(dtv))
summary_k_unif <- dtf_unif %>% group_by(k) %>% summarise(mean_dtv = mean(dtv))
summary_k_random <- dtf_random %>% group_by(k) %>% summarise(mean_dtv = mean(dtv))

dtf %>% 
  ggplot(aes(y = dtv, x = k)) +
  geom_jitter(aes(col = init), size = 3, alpha = 0.6) +
  geom_line(data = summary_k, aes(x = k, y = mean_dtv, group = 1), size = 1.5, linetype = 'dashed', alpha = 0.6) +
  geom_line(data = summary_k_mean, aes(x = k, y = mean_dtv, group = 1), size = 1.5, col = 'red', alpha = 0.6) +
  geom_line(data = summary_k_random, aes(x = k, y = mean_dtv, group = 1), size = 1.5, col = 'navy', alpha = 0.6) +
  geom_line(data = summary_k_unif, aes(x = k, y = mean_dtv, group = 1), size = 1.5, col = 'orange', alpha = 0.6) +
  labs(title = "Dtv vs number of observations", 
       x ="number of observations", y = "", subtitle = 'Grouped by initialization method') +
  scale_color_manual(values=c("red", "navy", "orange")) +
  theme_bw() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
