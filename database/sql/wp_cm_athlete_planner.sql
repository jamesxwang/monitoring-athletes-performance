-- phpMyAdmin SQL Dump
-- version 4.9.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Apr 29, 2021 at 11:35 AM
-- Server version: 10.4.18-MariaDB
-- PHP Version: 7.4.18

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `dev`
--

-- --------------------------------------------------------

--
-- Table structure for table `wp_cm_athlete_planner`
--

CREATE TABLE IF NOT EXISTS `wp_cm_athlete_planner` (
  `id` mediumint(9) NOT NULL,
  `planner_id` mediumint(9) NOT NULL,
  `user_id` mediumint(9) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `wp_cm_athlete_planner`
--

INSERT INTO `wp_cm_athlete_planner` (`id`, `planner_id`, `user_id`) VALUES
(1, 20, 56),
(2, 12, 2443),
(3, 23, 7),
(4, 19, 2692),
(7, 1, 2690),
(9, 4, 2693);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `wp_cm_athlete_planner`
--
ALTER TABLE `wp_cm_athlete_planner`
  ADD PRIMARY KEY (`id`),
  ADD KEY `planner_id` (`planner_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `wp_cm_athlete_planner`
--
ALTER TABLE `wp_cm_athlete_planner`
  MODIFY `id` mediumint(9) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `wp_cm_athlete_planner`
--
ALTER TABLE `wp_cm_athlete_planner`
  ADD CONSTRAINT `wp_cm_athlete_planner_ibfk_1` FOREIGN KEY (`planner_id`) REFERENCES `wp_cm_planner` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
