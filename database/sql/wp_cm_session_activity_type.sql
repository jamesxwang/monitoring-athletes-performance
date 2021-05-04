-- phpMyAdmin SQL Dump
-- version 4.9.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Apr 29, 2021 at 11:33 AM
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
-- Table structure for table `wp_cm_session_activity_type`
--

CREATE TABLE IF NOT EXISTS `wp_cm_session_activity_type` (
  `id` mediumint(9) NOT NULL,
  `name` varchar(250) NOT NULL,
  `value` varchar(250) NOT NULL,
  `img_url` varchar(500) NOT NULL,
  `color` varchar(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `wp_cm_session_activity_type`
--

INSERT INTO `wp_cm_session_activity_type` (`id`, `name`, `value`, `img_url`, `color`) VALUES
(1, 'Swimming', 'swimming', '/wp-content/plugins/coaching-mate-2/images/swim.png', '#0000FF'),
(2, 'Running', 'running', '/wp-content/plugins/coaching-mate-2/images/run.png', '#00CC00'),
(3, 'Lifting', 'weight', '/wp-content/plugins/coaching-mate-2/images/strength.png', '#FF9900'),
(4, 'Cycling', 'cycling', '/wp-content/plugins/coaching-mate-2/images/bike.png', '#FF0000'),
(5, 'Flexibility', 'flexibility', '/wp-content/plugins/coaching-mate-2/images/flexibility-2.png', '#663399'),
(6, 'Note', 'note', '/wp-content/plugins/coaching-mate-2/images/Note.png', '#000000'),
(7, 'Walk', 'walk', '/wp-content/plugins/coaching-mate-2/images/walk.png', '#00CC00'),
(8, 'Recovery', 'recovery', '/wp-content/plugins/coaching-mate-2/images/recovery.png', '#663399'),
(9, 'Aquathon', 'aquathon', '/wp-content/plugins/coaching-mate-2/images/Aquathon.png', '00CCFF'),
(10, 'Triathlon', 'triathlon', '/wp-content/plugins/coaching-mate-2/images/Triathlon-1.png', '#000066'),
(11, 'Circuit + Equipment', 'circuit-quipment', '', ''),
(12, 'Cycling-Stationary', 'cycling-stationary', '', ''),
(13, 'Duathlon', 'duathlon', '/wp-content/plugins/coaching-mate-2/images/Duathlon.png', '#990000');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `wp_cm_session_activity_type`
--
ALTER TABLE `wp_cm_session_activity_type`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `wp_cm_session_activity_type`
--
ALTER TABLE `wp_cm_session_activity_type`
  MODIFY `id` mediumint(9) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=14;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
