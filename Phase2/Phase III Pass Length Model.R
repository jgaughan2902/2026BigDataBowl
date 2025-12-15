# Libraries
library(tidyverse)
library(randomForest)
library(caret)

# <br>
#### Section 1.1: Importing the input (pre-pass) data

path = "C:/All class folders/Fall 2025/Sports Analytics/Final Project Phase 2/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"

input_files = list.files(
  path = path,
  pattern = "^input_.*\\.csv$",
  full.names = TRUE,
  ignore.case = TRUE,
  recursive = TRUE
)

input = input_files %>%
  map_dfr(~read_csv(.x, show_col_types = F)) %>%
  clean_names()

# <br>
#### Section 1.2: Importing the output (post-pass) data
output_files = list.files(
  path = path,
  pattern = "^output_.*\\.csv$",
  full.names = TRUE,
  ignore.case = TRUE,
  recursive = TRUE
)

output = output_files %>%
  map_dfr(~read_csv(.x, show_col_types = F))

# <br>
#### Section 1.3: Importing the supplementary data
supp = read_csv("C:/All class folders/Fall 2025/Sports Analytics/Final Project Phase 2/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv")

# -------------------------------
# Step 2: Player roles and trajectories
# -------------------------------
player_roles <- input %>%
  select(game_id, play_id, nfl_id, player_side, player_role, player_name, player_position) %>%
  distinct()

out_tagged <- output %>%
  left_join(player_roles, by = c("game_id", "play_id", "nfl_id"))

receiver_traj <- out_tagged %>%
  filter(player_side == "Offense", player_role %in% c("Receiver", "Targeted Receiver")) %>%
  select(game_id, play_id, frame_id, nfl_id, x, y) %>%
  rename(rec_nfl_id = nfl_id, rec_x = x, rec_y = y)

defenders_traj <- out_tagged %>%
  filter(player_side == "Defense") %>%
  select(game_id, play_id, frame_id, nfl_id, x, y) %>%
  rename(def_nfl_id = nfl_id, def_x = x, def_y = y)

# -------------------------------
# Step 3: Receiver–Defender primary matchups
# -------------------------------
pairwise <- receiver_traj %>%
  left_join(defenders_traj, by = c("game_id", "play_id", "frame_id")) %>%
  mutate(dist_to_receiver = sqrt((def_x - rec_x)^2 + (def_y - rec_y)^2))

# Keep only the nearest defender per receiver per frame
receiver_defender_pairs <- pairwise %>%
  group_by(game_id, play_id, frame_id, rec_nfl_id) %>%
  slice_min(order_by = dist_to_receiver, with_ties = FALSE) %>%
  ungroup()

# -------------------------------
# Step 4: Attach team and coverage info
# -------------------------------
game_teams <- supp %>%
  select(game_id, play_id, possession_team, home_team_abbr, visitor_team_abbr,
         team_coverage_man_zone, team_coverage_type) %>%
  distinct()

player_info <- player_roles %>%
  left_join(game_teams, by = c("game_id", "play_id")) %>%
  mutate(team = case_when(
    player_side == "Offense" & possession_team == home_team_abbr ~ home_team_abbr,
    player_side == "Offense" & possession_team == visitor_team_abbr ~ visitor_team_abbr,
    player_side == "Defense" & possession_team == home_team_abbr ~ visitor_team_abbr,
    player_side == "Defense" & possession_team == visitor_team_abbr ~ home_team_abbr,
    TRUE ~ NA_character_
  )) %>%
  select(game_id, play_id, nfl_id, player_name, player_position, team) %>%
  distinct()

receiver_defender_pairs <- receiver_defender_pairs %>%
  left_join(player_info, by = c("game_id", "play_id", "rec_nfl_id" = "nfl_id")) %>%
  rename(receiver_name = player_name, receiver_position = player_position, receiver_team = team) %>%
  left_join(player_info, by = c("game_id", "play_id", "def_nfl_id" = "nfl_id")) %>%
  rename(defender_name = player_name, defender_position = player_position, defender_team = team) %>%
  left_join(game_teams, by = c("game_id", "play_id"))

# -------------------------------
# Step 5: Coverage success thresholds
# -------------------------------
avg_man_separation <- receiver_defender_pairs %>%
  filter(team_coverage_man_zone == "MAN_COVERAGE") %>%
  summarise(avg_sep = mean(dist_to_receiver, na.rm = TRUE)) %>%
  pull(avg_sep)

avg_zone_separation <- receiver_defender_pairs %>%
  filter(team_coverage_man_zone == "ZONE_COVERAGE") %>%
  summarise(avg_sep = mean(dist_to_receiver, na.rm = TRUE)) %>%
  pull(avg_sep)

receiver_defender_pairs <- receiver_defender_pairs %>%
  mutate(good_coverage = case_when(
    team_coverage_man_zone == "MAN_COVERAGE" & dist_to_receiver <= avg_man_separation ~ 1,
    team_coverage_man_zone == "ZONE_COVERAGE" & dist_to_receiver <= avg_zone_separation ~ 1,
    TRUE ~ 0
  ))

# -------------------------------
# Step 6: Summarise coverage success at matchup level
# -------------------------------
coverage_summary <- receiver_defender_pairs %>%
  group_by(game_id, play_id, rec_nfl_id, def_nfl_id,
           receiver_name, defender_name, receiver_team, defender_team,
           team_coverage_man_zone, team_coverage_type) %>%
  summarise(
    coverage_success_pct = mean(good_coverage, na.rm = TRUE),
    frames_per_play = n(),
    .groups = "drop"
  )

# -------------------------------
# Step 7: Add pass info, yardline, and clock
# -------------------------------
supp <- supp %>%
  mutate(
    quarter_seconds = as.numeric(substr(game_clock, 1, 2)) * 60 +
      as.numeric(substr(game_clock, 4, 5)),
    time_remaining_game = (4 - quarter) * 900 + quarter_seconds
  )

pass_info <- supp %>%
  select(game_id, play_id, pass_length, down, yards_to_go, pass_result,
         time_remaining_game) %>%
  distinct()

coverage_summary <- coverage_summary %>%
  left_join(pass_info, by = c("game_id", "play_id"))

yardline <- input %>%
  select(game_id, play_id, absolute_yardline_number) %>%
  distinct()

coverage_summary <- coverage_summary %>%
  left_join(yardline, by = c("game_id", "play_id"))

# -------------------------------
# Step 8: Bin pass length
# -------------------------------
coverage_summary <- coverage_summary %>%
  mutate(pass_length_bin = cut(pass_length,
                               breaks = c(-Inf, 3, 10, Inf),
                               labels = c("<=3", "4-10", "10+"),
                               right = TRUE))

coverage_summary_clean <- coverage_summary %>%
  filter(!is.na(team_coverage_type),
         !is.na(def_nfl_id),
         !is.na(pass_length_bin))

# -------------------------------
# Step 9: Train/Test split
# -------------------------------
set.seed(123)
train_idx <- sample(seq_len(nrow(coverage_summary_clean)), size = 0.7 * nrow(coverage_summary_clean))
train_data <- coverage_summary_clean[train_idx, ]
test_data  <- coverage_summary_clean[-train_idx, ]

# -------------------------------
# Step 10: Random Forest model
# -------------------------------
rf_model <- randomForest(
  pass_length_bin ~ coverage_success_pct + down + yards_to_go +
    team_coverage_type + receiver_name + defender_name +
    receiver_team + defender_team + absolute_yardline_number +
    time_remaining_game,
  data = train_data,
  ntree = 500,
  importance = TRUE,
  na.action = na.omit
)

model = glm(pass_length_bin ~ coverage_success_pct + down + yards_to_go +
     team_coverage_type + receiver_name + defender_name +
     receiver_team + defender_team + absolute_yardline_number +
     time_remaining_game, family = "binomial",
   data = train_data)

predictions <- predict(model, newdata = new_data_frame, type = "response")

# -------------------------------
# Step 11: Evaluate accuracy
# -------------------------------
pred_classes <- predict(rf_model, newdata = test_data, type = "class")
confusionMatrix(pred_classes, test_data$pass_length_bin)
varImpPlot(rf_model)


# -------------------------------
# Step 12: Build Out Probabilities
# -------------------------------

rf_model <- randomForest(
  pass_length_bin ~ coverage_success_pct + down + yards_to_go +
    team_coverage_type + receiver_name + receiver_team + defender_team + absolute_yardline_number +
    time_remaining_game,
  data = coverage_summary_clean,
  ntree = 500,
  mtry = 3,
  importance = TRUE,
  na.action = na.omit
)

newdata <- data.frame(
  coverage_success_pct = .75,
  down = 3,
  yards_to_go = 8,
  team_coverage_type = "COVER_2_ZONE",
  receiver_name = "Amon-Ra St. Brown",
  receiver_team = "DET",
  defender_team = "GB",
  absolute_yardline_number = 88,
  time_remaining_game = 12
)

predict(rf_model, newdata, type = "prob")
