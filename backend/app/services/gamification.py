"""
Gamification Service - Citizen Engagement System
Implements points, badges, leaderboards, and blockchain integration
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib

class BadgeType(Enum):
    REPORTER = "reporter"
    VERIFIER = "verifier"
    COMMUNITY_GUARDIAN = "community_guardian"
    ECO_WARRIOR = "eco_warrior"
    DATA_CONTRIBUTOR = "data_contributor"
    PREVENTION_CHAMPION = "prevention_champion"

class AchievementTier(Enum):
    BRONZE = "bronze"
    SILVER = "silver" 
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"

@dataclass
class Badge:
    id: str
    name: str
    description: str
    badge_type: BadgeType
    tier: AchievementTier
    icon: str
    points_reward: int
    requirements: Dict[str, Any]
    rarity: str  # common, rare, epic, legendary

@dataclass
class UserProfile:
    user_id: str
    username: str
    total_points: int
    level: int
    badges: List[str]  # badge IDs
    reports_submitted: int
    reports_verified: int
    streak_days: int
    last_active: datetime
    location: Optional[Dict[str, float]]
    blockchain_wallet: Optional[str] = None

@dataclass
class Achievement:
    user_id: str
    badge_id: str
    earned_date: datetime
    points_awarded: int
    blockchain_hash: Optional[str] = None

@dataclass
class LeaderboardEntry:
    rank: int
    user_id: str
    username: str
    points: int
    level: int
    badges_count: int
    recent_activity: str

class GamificationEngine:
    """
    Gamification engine for citizen engagement
    Manages points, badges, leaderboards, and blockchain integration
    """
    
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}
        self.badges: Dict[str, Badge] = {}
        self.achievements: List[Achievement] = []
        
        # Initialize badge system
        self._initialize_badges()
        
        # Simulated blockchain integration
        self.blockchain_enabled = True
        self.blockchain_rewards = []
    
    def _initialize_badges(self):
        """Initialize all available badges"""
        
        badges_data = [
            # Reporter Badges
            {
                "id": "first_reporter",
                "name": "First Report",
                "description": "Submit your first dengue breeding site report",
                "badge_type": BadgeType.REPORTER,
                "tier": AchievementTier.BRONZE,
                "icon": "🔍",
                "points_reward": 50,
                "requirements": {"reports_submitted": 1},
                "rarity": "common"
            },
            {
                "id": "active_reporter",
                "name": "Active Reporter",
                "description": "Submit 10 verified reports",
                "badge_type": BadgeType.REPORTER,
                "tier": AchievementTier.SILVER,
                "icon": "📋",
                "points_reward": 200,
                "requirements": {"verified_reports": 10},
                "rarity": "rare"
            },
            {
                "id": "super_reporter",
                "name": "Super Reporter",
                "description": "Submit 50 verified reports",
                "badge_type": BadgeType.REPORTER,
                "tier": AchievementTier.GOLD,
                "icon": "⭐",
                "points_reward": 500,
                "requirements": {"verified_reports": 50},
                "rarity": "epic"
            },
            
            # Community Badges
            {
                "id": "community_guardian",
                "name": "Community Guardian",
                "description": "Help verify 25 community reports",
                "badge_type": BadgeType.COMMUNITY_GUARDIAN,
                "tier": AchievementTier.SILVER,
                "icon": "🛡️",
                "points_reward": 300,
                "requirements": {"reports_verified": 25},
                "rarity": "rare"
            },
            {
                "id": "eco_warrior",
                "name": "Eco Warrior",
                "description": "Maintain a 30-day reporting streak",
                "badge_type": BadgeType.ECO_WARRIOR,
                "tier": AchievementTier.GOLD,
                "icon": "🌱",
                "points_reward": 750,
                "requirements": {"streak_days": 30},
                "rarity": "epic"
            },
            {
                "id": "prevention_champion",
                "name": "Prevention Champion",
                "description": "Reach level 10 and earn 2000 points",
                "badge_type": BadgeType.PREVENTION_CHAMPION,
                "tier": AchievementTier.PLATINUM,
                "icon": "🏆",
                "points_reward": 1000,
                "requirements": {"level": 10, "total_points": 2000},
                "rarity": "legendary"
            },
            
            # Special Achievement
            {
                "id": "blockchain_pioneer",
                "name": "Blockchain Pioneer",
                "description": "First to earn blockchain-verified rewards",
                "badge_type": BadgeType.DATA_CONTRIBUTOR,
                "tier": AchievementTier.DIAMOND,
                "icon": "⛓️",
                "points_reward": 1500,
                "requirements": {"blockchain_transactions": 1},
                "rarity": "legendary"
            }
        ]
        
        for badge_data in badges_data:
            badge = Badge(**badge_data)
            self.badges[badge.id] = badge
    
    def register_user(self, user_id: str, username: str, location: Optional[Dict[str, float]] = None) -> UserProfile:
        """Register a new user in the gamification system"""
        
        if user_id in self.users:
            return self.users[user_id]
        
        user = UserProfile(
            user_id=user_id,
            username=username,
            total_points=0,
            level=1,
            badges=[],
            reports_submitted=0,
            reports_verified=0,
            streak_days=0,
            last_active=datetime.now(),
            location=location
        )
        
        self.users[user_id] = user
        
        # Award welcome bonus
        self.award_points(user_id, 25, "Welcome bonus")
        
        return user
    
    def submit_report(self, user_id: str, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a submitted report and award points/badges"""
        
        if user_id not in self.users:
            return {"error": "User not registered"}
        
        user = self.users[user_id]
        user.reports_submitted += 1
        user.last_active = datetime.now()
        
        # Base points for report submission
        base_points = 25
        
        # Quality bonus based on report data
        quality_multiplier = 1.0
        
        # Check for photo upload
        if report_data.get("has_photo", False):
            quality_multiplier += 0.3
            
        # Check for GPS coordinates
        if report_data.get("location"):
            quality_multiplier += 0.2
            
        # Check for detailed description
        description = report_data.get("description", "")
        if len(description) > 50:
            quality_multiplier += 0.2
        
        # Award points
        points_earned = int(base_points * quality_multiplier)
        self.award_points(user_id, points_earned, "Report submission")
        
        # Update streak
        self._update_streak(user_id)
        
        # Check for badge achievements
        new_badges = self._check_badge_achievements(user_id)
        
        return {
            "points_earned": points_earned,
            "total_points": user.total_points,
            "level": user.level,
            "new_badges": [self.badges[badge_id].name for badge_id in new_badges],
            "quality_multiplier": quality_multiplier,
            "streak_days": user.streak_days
        }
    
    def verify_report(self, user_id: str, report_id: str, is_verified: bool) -> Dict[str, Any]:
        """Process report verification and award points"""
        
        if user_id not in self.users:
            return {"error": "User not registered"}
        
        user = self.users[user_id]
        user.reports_verified += 1
        user.last_active = datetime.now()
        
        # Points for verification activity
        verification_points = 15 if is_verified else 10
        self.award_points(user_id, verification_points, "Report verification")
        
        # Check for badge achievements
        new_badges = self._check_badge_achievements(user_id)
        
        return {
            "points_earned": verification_points,
            "total_points": user.total_points,
            "level": user.level,
            "new_badges": [self.badges[badge_id].name for badge_id in new_badges]
        }
    
    def award_points(self, user_id: str, points: int, reason: str) -> bool:
        """Award points to a user and update level"""
        
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        user.total_points += points
        
        # Update level (every 100 points = 1 level)
        new_level = min(50, max(1, 1 + (user.total_points // 100)))
        
        level_up = new_level > user.level
        user.level = new_level
        
        # Level up bonus
        if level_up:
            bonus_points = new_level * 10
            user.total_points += bonus_points
            
            # Blockchain reward for major milestones
            if new_level % 5 == 0 and self.blockchain_enabled:
                self._issue_blockchain_reward(user_id, new_level)
        
        return level_up
    
    def _update_streak(self, user_id: str):
        """Update user's activity streak"""
        
        user = self.users[user_id]
        now = datetime.now()
        
        # Check if last active was yesterday
        last_active_date = user.last_active.date()
        yesterday = (now - timedelta(days=1)).date()
        today = now.date()
        
        if last_active_date == yesterday:
            # Continue streak
            user.streak_days += 1
        elif last_active_date == today:
            # Same day, no change to streak
            pass
        else:
            # Streak broken, reset
            user.streak_days = 1
        
        user.last_active = now
        
        # Streak milestone rewards
        if user.streak_days in [7, 14, 30, 60, 90]:
            bonus_points = user.streak_days * 2
            self.award_points(user_id, bonus_points, f"Streak milestone: {user.streak_days} days")
    
    def _check_badge_achievements(self, user_id: str) -> List[str]:
        """Check and award any newly earned badges"""
        
        if user_id not in self.users:
            return []
        
        user = self.users[user_id]
        new_badges = []
        
        for badge_id, badge in self.badges.items():
            # Skip if already earned
            if badge_id in user.badges:
                continue
            
            # Check requirements
            earned = True
            for req_key, req_value in badge.requirements.items():
                
                if req_key == "reports_submitted" and user.reports_submitted < req_value:
                    earned = False
                elif req_key == "verified_reports" and user.reports_verified < req_value:
                    earned = False
                elif req_key == "reports_verified" and user.reports_verified < req_value:
                    earned = False
                elif req_key == "streak_days" and user.streak_days < req_value:
                    earned = False
                elif req_key == "level" and user.level < req_value:
                    earned = False
                elif req_key == "total_points" and user.total_points < req_value:
                    earned = False
                elif req_key == "blockchain_transactions":
                    # Check blockchain activity (simplified)
                    blockchain_count = len([r for r in self.blockchain_rewards if r["user_id"] == user_id])
                    if blockchain_count < req_value:
                        earned = False
            
            if earned:
                # Award badge
                user.badges.append(badge_id)
                new_badges.append(badge_id)
                
                # Award points
                user.total_points += badge.points_reward
                
                # Record achievement
                achievement = Achievement(
                    user_id=user_id,
                    badge_id=badge_id,
                    earned_date=datetime.now(),
                    points_awarded=badge.points_reward
                )
                
                self.achievements.append(achievement)
                
                # Blockchain verification for rare badges
                if badge.rarity in ["epic", "legendary"] and self.blockchain_enabled:
                    achievement.blockchain_hash = self._create_blockchain_achievement(user_id, badge_id)
        
        return new_badges
    
    def _issue_blockchain_reward(self, user_id: str, level: int):
        """Issue blockchain-based reward token"""
        
        reward_data = {
            "user_id": user_id,
            "level": level,
            "timestamp": datetime.now().isoformat(),
            "type": "level_milestone",
            "value": level * 10
        }
        
        # Simulate blockchain transaction
        reward_hash = hashlib.sha256(
            json.dumps(reward_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        reward_data["blockchain_hash"] = reward_hash
        self.blockchain_rewards.append(reward_data)
        
        return reward_hash
    
    def _create_blockchain_achievement(self, user_id: str, badge_id: str) -> str:
        """Create blockchain-verified achievement record"""
        
        achievement_data = {
            "user_id": user_id,
            "badge_id": badge_id,
            "timestamp": datetime.now().isoformat(),
            "type": "badge_achievement"
        }
        
        # Simulate blockchain hash
        achievement_hash = hashlib.sha256(
            json.dumps(achievement_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        return achievement_hash
    
    def get_leaderboard(self, limit: int = 10, location_filter: Optional[Dict[str, float]] = None) -> List[LeaderboardEntry]:
        """Get leaderboard rankings"""
        
        # Filter users by location if specified
        users_list = list(self.users.values())
        
        if location_filter:
            # Simple distance-based filtering (simplified)
            filtered_users = []
            for user in users_list:
                if user.location:
                    # Simple distance calculation (for demo)
                    lat_diff = abs(user.location["lat"] - location_filter["lat"])
                    lng_diff = abs(user.location["lng"] - location_filter["lng"])
                    distance = (lat_diff ** 2 + lng_diff ** 2) ** 0.5
                    
                    if distance < 0.1:  # ~11km radius
                        filtered_users.append(user)
            
            users_list = filtered_users
        
        # Sort by points
        sorted_users = sorted(users_list, key=lambda u: u.total_points, reverse=True)
        
        # Create leaderboard entries
        leaderboard = []
        for i, user in enumerate(sorted_users[:limit]):
            
            # Determine recent activity
            days_since_active = (datetime.now() - user.last_active).days
            if days_since_active == 0:
                recent_activity = "Active today"
            elif days_since_active == 1:
                recent_activity = "Active yesterday"
            elif days_since_active < 7:
                recent_activity = f"Active {days_since_active} days ago"
            else:
                recent_activity = "Inactive"
            
            entry = LeaderboardEntry(
                rank=i + 1,
                user_id=user.user_id,
                username=user.username,
                points=user.total_points,
                level=user.level,
                badges_count=len(user.badges),
                recent_activity=recent_activity
            )
            
            leaderboard.append(entry)
        
        return leaderboard
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get complete user profile with achievements"""
        
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        
        # Get user's badges with details
        user_badges = []
        for badge_id in user.badges:
            if badge_id in self.badges:
                badge = self.badges[badge_id]
                user_badges.append({
                    "id": badge.id,
                    "name": badge.name,
                    "description": badge.description,
                    "tier": badge.tier.value,
                    "icon": badge.icon,
                    "rarity": badge.rarity
                })
        
        # Get achievements with blockchain verification
        user_achievements = [
            {
                "badge_name": self.badges[a.badge_id].name,
                "earned_date": a.earned_date.isoformat(),
                "points_awarded": a.points_awarded,
                "blockchain_verified": a.blockchain_hash is not None,
                "blockchain_hash": a.blockchain_hash
            }
            for a in self.achievements
            if a.user_id == user_id
        ]
        
        # Calculate next level progress
        points_for_next_level = (user.level * 100) - user.total_points
        progress_to_next_level = max(0, min(100, 
            ((user.total_points % 100) / 100) * 100
        ))
        
        return {
            "profile": asdict(user),
            "badges": user_badges,
            "achievements": user_achievements,
            "next_level_progress": {
                "current_level": user.level,
                "progress_percentage": progress_to_next_level,
                "points_needed": max(0, points_for_next_level)
            },
            "blockchain_rewards": [
                r for r in self.blockchain_rewards 
                if r["user_id"] == user_id
            ]
        }
    
    def get_available_badges(self) -> List[Dict[str, Any]]:
        """Get all available badges for display"""
        
        return [
            {
                "id": badge.id,
                "name": badge.name,
                "description": badge.description,
                "tier": badge.tier.value,
                "icon": badge.icon,
                "points_reward": badge.points_reward,
                "requirements": badge.requirements,
                "rarity": badge.rarity
            }
            for badge in self.badges.values()
        ]
    
    def get_gamification_stats(self) -> Dict[str, Any]:
        """Get overall gamification system statistics"""
        
        total_users = len(self.users)
        total_points_awarded = sum(user.total_points for user in self.users.values())
        total_badges_earned = sum(len(user.badges) for user in self.users.values())
        active_users = sum(1 for user in self.users.values() 
                          if (datetime.now() - user.last_active).days < 7)
        
        return {
            "total_users": total_users,
            "active_users_7_days": active_users,
            "total_points_awarded": total_points_awarded,
            "total_badges_earned": total_badges_earned,
            "total_achievements": len(self.achievements),
            "blockchain_rewards_issued": len(self.blockchain_rewards),
            "average_user_level": sum(user.level for user in self.users.values()) / max(1, total_users),
            "top_user_points": max((user.total_points for user in self.users.values()), default=0)
        }

# Create global instance
gamification_engine = GamificationEngine()