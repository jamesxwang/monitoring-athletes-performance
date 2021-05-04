-- phpMyAdmin SQL Dump
-- version 4.9.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Apr 29, 2021 at 11:36 AM
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
-- Table structure for table `wp_cm_planner_program`
--

CREATE TABLE IF NOT EXISTS `wp_cm_planner_program` (
  `id` mediumint(9) NOT NULL,
  `planner_id` mediumint(9) NOT NULL,
  `program_id` mediumint(9) NOT NULL,
  `associate_program_start_date` date DEFAULT NULL,
  `layer` smallint(6) NOT NULL DEFAULT 1,
  `athlete_id` bigint(20) NOT NULL DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `wp_cm_planner_program`
--

INSERT INTO `wp_cm_planner_program` (`id`, `planner_id`, `program_id`, `associate_program_start_date`, `layer`, `athlete_id`) VALUES
(1, 13, 605, '2018-04-30', 1, 0),
(2, 13, 623, '2018-05-28', 1, 0),
(3, 13, 649, '2018-07-02', 1, 0),
(9, 15, 747, '2018-11-19', 1, 0),
(13, 23, 761, '2018-11-26', 1, 0),
(15, 26, 747, '2018-11-19', 1, 0),
(16, 27, 747, '2018-10-29', 1, 0),
(18, 20, 623, '2018-05-28', 1, 0),
(19, 20, 649, '2018-07-02', 1, 0),
(20, 20, 677, '2018-07-30', 1, 0),
(21, 20, 703, '2018-08-27', 1, 0),
(22, 20, 727, '2018-10-01', 1, 0),
(23, 20, 747, '2018-10-29', 1, 0),
(24, 20, 605, '2018-04-30', 1, 0),
(30, 20, 775, '2019-06-17', 1, 0),
(31, 13, 677, '2018-07-30', 1, 0),
(32, 13, 703, '2018-08-27', 1, 0),
(33, 13, 727, '2018-10-01', 1, 0),
(34, 13, 746, '2018-10-29', 1, 0),
(35, 13, 775, '2018-11-26', 1, 0),
(36, 10, 787, '2018-02-26', 1, 0),
(38, 12, 789, '2019-04-22', 1, 0),
(41, 23, 791, '2018-12-31', 1, 0),
(44, 20, 801, '2019-02-25', 1, 0),
(46, 19, 126, '2019-05-13', 1, 0),
(47, 2, 498, '2017-07-03', 1, 0),
(51, 1, 498, '2018-04-16', 1, 0),
(55, 1, 791, '2018-01-08', 1, 0),
(56, 12, 787, '2018-06-11', 2, 0),
(57, 12, 791, '2018-08-20', 1, 0),
(58, 12, 568, '2019-04-29', 1, 0),
(59, 30, 84, '2019-07-08', 2, 0),
(60, 30, 102, '2019-09-09', 2, 0),
(61, 31, 59, '2019-07-01', 3, 0),
(62, 31, 205, '2019-11-04', 3, 0),
(63, 31, 204, '2019-06-10', 2, 0),
(64, 31, 207, '2019-07-15', 3, 0),
(65, 31, 816, '2019-07-08', 2, 0),
(66, 12, 59, '2019-06-24', 2, 0),
(67, 12, 99, '2020-01-06', 3, 0),
(68, 12, 84, '2020-03-23', 1, 0),
(69, 12, 196, '2019-10-28', 2, 0),
(70, 12, 137, '2020-01-27', 2, 0),
(71, 2, 59, '2020-03-23', 2, 0),
(72, 7, 59, '2018-03-05', 1, 0),
(73, 32, 605, '2019-05-06', 1, 0),
(74, 32, 623, '2019-06-03', 1, 0),
(75, 32, 649, '2019-07-08', 1, 0),
(76, 32, 676, '2019-08-05', 1, 0),
(77, 32, 702, '2019-09-02', 1, 0),
(78, 32, 726, '2019-10-07', 1, 0),
(79, 32, 774, '2019-11-04', 1, 0),
(81, 1, 59, '2017-09-25', 2, 0),
(82, 12, 104, '2019-04-29', 4, 2687),
(83, 12, 136, '2019-09-30', 4, 2687),
(84, 12, 139, '2019-06-03', 4, 2687),
(85, 12, 206, '2019-02-25', 1, 0),
(86, 12, 210, '2019-07-15', 4, 2687),
(87, 12, 184, '2018-06-04', 4, 2687),
(88, 12, 169, '2018-07-30', 4, 2687),
(89, 12, 762, '2018-07-16', 4, 2687),
(90, 31, 84, '2019-06-24', 4, 2693),
(91, 31, 762, '2020-01-13', 4, 2693),
(93, 30, 59, '2019-05-06', 1, 0),
(94, 31, 84, '2020-03-16', 2, 0),
(97, 35, 59, '2019-09-02', 1, 0),
(98, 35, 84, '2019-12-02', 2, 0),
(99, 35, 101, '2019-08-12', 3, 0),
(100, 35, 818, '2019-11-11', 3, 0),
(102, 1, 98, '2017-11-06', 2, 0),
(103, 1, 84, '2017-05-08', 2, 0),
(110, 1, 81, '2018-01-22', 3, 0);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `wp_cm_planner_program`
--
ALTER TABLE `wp_cm_planner_program`
  ADD PRIMARY KEY (`id`),
  ADD KEY `program_id` (`program_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `wp_cm_planner_program`
--
ALTER TABLE `wp_cm_planner_program`
  MODIFY `id` mediumint(9) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=111;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
