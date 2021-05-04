-- phpMyAdmin SQL Dump
-- version 4.9.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Apr 29, 2021 at 11:34 AM
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
-- Table structure for table `wp_cm_planner`
--

CREATE TABLE IF NOT EXISTS `wp_cm_planner` (
  `id` mediumint(9) NOT NULL,
  `title` varchar(250) NOT NULL,
  `starting_date` date NOT NULL,
  `end_interval` int(11) NOT NULL DEFAULT 12,
  `display_countdown` varchar(3) NOT NULL DEFAULT 'yes',
  `reverse_countdown` varchar(3) NOT NULL DEFAULT 'yes',
  `tc_start_date` date NOT NULL,
  `tc_interval` smallint(6) NOT NULL DEFAULT 2,
  `added_by` bigint(20) UNSIGNED NOT NULL DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `wp_cm_planner`
--

INSERT INTO `wp_cm_planner` (`id`, `title`, `starting_date`, `end_interval`, `display_countdown`, `reverse_countdown`, `tc_start_date`, `tc_interval`, `added_by`) VALUES
(1, 'Short Course April 2017', '2017-04-17', 20, 'yes', 'yes', '2018-09-03', 3, 5),
(2, 'Test Planner', '2016-07-01', 12, 'no', 'yes', '2019-07-01', 2, 5),
(3, 'Test', '2017-05-01', 12, 'no', 'yes', '0000-00-00', 2, 5),
(4, 'Test Planner 2', '2017-05-01', 12, 'no', 'yes', '0000-00-00', 2, 5),
(5, 'Swim Planner', '2017-06-27', 12, 'no', 'no', '0000-00-00', 3, 5),
(6, 'Test Planner-Ollie', '2017-07-03', 12, 'no', 'yes', '0000-00-00', 2, 5),
(7, 'Ollie Test 1', '2017-07-03', 23, 'yes', 'yes', '2019-07-03', 2, 5),
(8, 'Ollie Test 2', '2017-07-03', 12, 'no', 'yes', '0000-00-00', 2, 5),
(9, 'Planner test 3', '2017-07-31', 12, 'no', 'yes', '0000-00-00', 2, 5),
(10, 'Strength Phase 1', '2017-08-27', 12, 'no', 'yes', '0000-00-00', 2, 5),
(11, 'Ironman Busso 2017', '2017-05-01', 12, 'no', 'yes', '0000-00-00', 2, 5),
(12, 'Short Course (INT) 2018', '2018-06-26', 23, 'yes', 'no', '2019-06-27', 2, 5),
(13, 'Michelle LC Planner 2018', '2018-04-30', 20, 'yes', 'yes', '2018-04-16', 2, 5),
(14, 'Bond IM program ', '2018-08-13', 4, 'no', 'yes', '0000-00-00', 2, 5),
(15, 'Test Planner Ejaz', '2018-08-01', 12, 'yes', 'yes', '2018-10-01', 2, 5),
(16, 'New Planner Test EJaz', '2017-08-01', 23, 'yes', 'yes', '2018-10-21', 3, 5),
(17, 'Mbt Test Planner', '2017-07-01', 20, 'yes', 'yes', '2018-09-30', 3, 5),
(18, 'Graph Mbt', '2017-11-01', 20, 'yes', 'yes', '2018-10-01', 3, 5),
(19, 'Ironman Advanced 2018/19', '2018-05-06', 18, 'yes', 'no', '2018-05-06', 2, 5),
(20, 'Ironman Intermediate 2018/19', '2018-04-30', 18, 'yes', 'yes', '2018-04-23', 2, 5),
(21, 'Ironman First Timer 2018/19', '2018-05-06', 15, 'yes', 'yes', '2018-05-06', 2, 5),
(22, 'Half Ironman Advanced 2018/19', '2018-05-06', 15, 'yes', 'yes', '2018-05-06', 2, 5),
(23, 'Half Ironman Intermediate 2018/19', '2018-05-06', 15, 'yes', 'yes', '2018-05-06', 2, 5),
(24, 'Half Ironman First Timer 2018/19', '2018-05-06', 15, 'yes', 'yes', '2018-05-06', 2, 5),
(25, 'Duplicate Planner Test', '2018-08-01', 12, 'yes', 'yes', '2018-10-01', 2, 5),
(26, 'Test Planner Ejaz Duplicate', '2018-08-01', 12, 'yes', 'yes', '2018-10-01', 2, 5),
(27, 'Test Planner Ejaz Duplicate 2', '2018-08-01', 7, 'yes', 'yes', '2018-10-01', 2, 5),
(28, 'Ishi planner 1', '2019-01-10', 23, 'yes', 'yes', '2019-01-02', 2, 5),
(29, 'ABC', '2019-02-06', 4, 'yes', 'yes', '2019-02-07', 2, 5),
(30, 'Basic Planner', '2019-05-30', 6, 'yes', 'yes', '2019-07-04', 3, 5),
(31, 'Testing Phase', '2019-06-25', 23, 'yes', 'no', '2019-07-15', 3, 5),
(32, 'Advanced Ironman', '2019-05-06', 12, 'yes', 'yes', '2019-05-13', 2, 5),
(33, 'Grid Planner', '2019-07-17', 10, 'yes', 'no', '2019-08-12', 3, 5),
(34, 'Demo1', '2019-08-20', 6, 'yes', 'yes', '2019-08-30', 2, 5),
(35, 'Sample Workout', '2019-08-31', 5, 'yes', 'yes', '2019-09-09', 2, 5),
(36, 'test again', '2019-12-03', 12, 'yes', 'yes', '2019-12-06', 2, 5);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `wp_cm_planner`
--
ALTER TABLE `wp_cm_planner`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `wp_cm_planner`
--
ALTER TABLE `wp_cm_planner`
  MODIFY `id` mediumint(9) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=37;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
