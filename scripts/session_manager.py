#!/usr/bin/env python3
"""
Session Manager for Research Demo Tools
Creates organized folder structure for each session with timestamps
"""

import os
import json
from datetime import datetime
from pathlib import Path
import shutil

class SessionManager:
    def __init__(self, base_dir="suite"):
        """Initialize session manager with base directory"""
        self.base_dir = Path(base_dir)
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"session_{self.session_timestamp}"
        self.session_dir = self.base_dir / self.session_name
        
        # Create session directory structure
        self.setup_session_directories()
        
        # Session metadata
        self.session_info = {
            'session_id': self.session_name,
            'start_time': datetime.now().isoformat(),
            'tools_used': [],
            'files_generated': [],
            'models_tested': [],
            'notes': []
        }
        
        print(f"📁 Session Manager initialized: {self.session_dir}")
        
    def setup_session_directories(self):
        """Create organized directory structure for the session"""
        
        # Main session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different types of outputs
        self.directories = {
            'images': self.session_dir / 'images',
            'graphs': self.session_dir / 'graphs',
            'logs': self.session_dir / 'logs',
            'reports': self.session_dir / 'reports',
            'videos': self.session_dir / 'videos',
            'data': self.session_dir / 'data',
            'configs': self.session_dir / 'configs',
            'models': self.session_dir / 'models',
            'temp': self.session_dir / 'temp'
        }
        
        # Create all subdirectories
        for dir_name, dir_path in self.directories.items():
            dir_path.mkdir(exist_ok=True)
            
        print(f"✅ Created session directories: {list(self.directories.keys())}")
        
    def get_session_path(self, category: str = None, filename: str = None):
        """Get path for saving files in the current session"""
        
        if category is None:
            return self.session_dir
            
        if category not in self.directories:
            # Create new category if it doesn't exist
            self.directories[category] = self.session_dir / category
            self.directories[category].mkdir(exist_ok=True)
            
        base_path = self.directories[category]
        
        if filename is None:
            return base_path
            
        # Add timestamp to filename if not already present
        if not any(ts in filename for ts in [self.session_timestamp[:8], self.session_timestamp]):
            name_parts = filename.split('.')
            if len(name_parts) > 1:
                base_name = '.'.join(name_parts[:-1])
                extension = name_parts[-1]
                filename = f"{base_name}_{self.session_timestamp}.{extension}"
            else:
                filename = f"{filename}_{self.session_timestamp}"
                
        return base_path / filename
    
    def save_image(self, fig, filename: str, category: str = 'images', **kwargs):
        """Save matplotlib figure with proper organization"""
        save_path = self.get_session_path(category, filename)
        
        # Default save parameters
        save_kwargs = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        save_kwargs.update(kwargs)
        
        fig.savefig(save_path, **save_kwargs)
        
        # Record in session info
        self.session_info['files_generated'].append({
            'type': 'image',
            'category': category,
            'filename': filename,
            'path': str(save_path),
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"💾 Saved image: {save_path}")
        return save_path
    
    def save_data(self, data, filename: str, category: str = 'data', format: str = 'json'):
        """Save data with proper organization"""
        save_path = self.get_session_path(category, filename)
        
        # Ensure correct extension
        if not save_path.suffix:
            save_path = save_path.with_suffix(f'.{format}')
        
        # Save based on format
        if format.lower() == 'json':
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format.lower() == 'txt':
            with open(save_path, 'w') as f:
                f.write(str(data))
        elif format.lower() == 'csv':
            # Assume pandas DataFrame or similar
            data.to_csv(save_path, index=False)
        
        # Record in session info
        self.session_info['files_generated'].append({
            'type': 'data',
            'category': category,
            'filename': filename,
            'path': str(save_path),
            'format': format,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"💾 Saved data: {save_path}")
        return save_path
    
    def save_log(self, log_content: str, filename: str = None, append: bool = False):
        """Save log content"""
        if filename is None:
            filename = f"session_log.txt"
            
        save_path = self.get_session_path('logs', filename)
        
        mode = 'a' if append else 'w'
        with open(save_path, mode) as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {log_content}\n")
        
        print(f"📝 Logged: {log_content[:50]}..." if len(log_content) > 50 else f"📝 Logged: {log_content}")
        return save_path
    
    def save_report(self, content: str, filename: str, format: str = 'txt'):
        """Save report with proper formatting"""
        save_path = self.get_session_path('reports', filename)
        
        # Ensure correct extension
        if not save_path.suffix:
            save_path = save_path.with_suffix(f'.{format}')
        
        # Add header with session info
        header = f"""
{'='*80}
RESEARCH SESSION REPORT
{'='*80}
Session ID: {self.session_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session Directory: {self.session_dir}
{'='*80}

"""
        
        with open(save_path, 'w') as f:
            f.write(header + content)
        
        # Record in session info
        self.session_info['files_generated'].append({
            'type': 'report',
            'category': 'reports',
            'filename': filename,
            'path': str(save_path),
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"📊 Saved report: {save_path}")
        return save_path
    
    def copy_file(self, source_path: str, category: str, new_filename: str = None):
        """Copy external file into session directory"""
        source = Path(source_path)
        
        if not source.exists():
            print(f"❌ Source file not found: {source}")
            return None
            
        filename = new_filename or source.name
        dest_path = self.get_session_path(category, filename)
        
        shutil.copy2(source, dest_path)
        
        # Record in session info
        self.session_info['files_generated'].append({
            'type': 'copied_file',
            'category': category,
            'filename': filename,
            'original_path': str(source),
            'path': str(dest_path),
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"📋 Copied file: {dest_path}")
        return dest_path
    
    def add_tool_usage(self, tool_name: str, details: str = ""):
        """Record tool usage in session"""
        self.session_info['tools_used'].append({
            'tool': tool_name,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_model_test(self, model_name: str, results: dict = None):
        """Record model testing in session"""
        self.session_info['models_tested'].append({
            'model': model_name,
            'results': results or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def add_note(self, note: str):
        """Add note to session"""
        self.session_info['notes'].append({
            'note': note,
            'timestamp': datetime.now().isoformat()
        })
        
        # Also save to log
        self.save_log(f"NOTE: {note}", append=True)
    
    def finalize_session(self):
        """Finalize session and save metadata"""
        
        self.session_info['end_time'] = datetime.now().isoformat()
        self.session_info['duration'] = str(datetime.now() - datetime.fromisoformat(self.session_info['start_time']))
        
        # Save session metadata
        metadata_path = self.session_dir / 'session_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.session_info, f, indent=2, default=str)
        
        # Create session summary
        summary = self.generate_session_summary()
        self.save_report(summary, 'session_summary.txt')
        
        print(f"🎯 Session finalized: {self.session_dir}")
        print(f"📁 Total files generated: {len(self.session_info['files_generated'])}")
        
        return self.session_dir
    
    def generate_session_summary(self):
        """Generate session summary text"""
        
        summary = f"""
SESSION SUMMARY
{'-'*40}

Tools Used: {len(self.session_info['tools_used'])}
{chr(10).join([f"  • {tool['tool']}: {tool['details']}" for tool in self.session_info['tools_used']])}

Models Tested: {len(self.session_info['models_tested'])}
{chr(10).join([f"  • {model['model']}" for model in self.session_info['models_tested']])}

Files Generated: {len(self.session_info['files_generated'])}
"""
        
        # Group files by category
        files_by_category = {}
        for file_info in self.session_info['files_generated']:
            category = file_info['category']
            if category not in files_by_category:
                files_by_category[category] = []
            files_by_category[category].append(file_info['filename'])
        
        for category, files in files_by_category.items():
            summary += f"\n{category.upper()}:\n"
            for filename in files:
                summary += f"  • {filename}\n"
        
        # Add notes
        if self.session_info['notes']:
            summary += f"\nNOTES:\n"
            for note_info in self.session_info['notes']:
                summary += f"  • {note_info['note']}\n"
        
        return summary
    
    def list_sessions(self, base_dir: str = None):
        """List all available sessions"""
        search_dir = Path(base_dir) if base_dir else self.base_dir
        
        if not search_dir.exists():
            print("No sessions found.")
            return []
        
        sessions = []
        for session_dir in search_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('session_'):
                metadata_file = session_dir / 'session_metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    sessions.append({
                        'name': session_dir.name,
                        'path': str(session_dir),
                        'start_time': metadata.get('start_time'),
                        'tools_used': len(metadata.get('tools_used', [])),
                        'files_generated': len(metadata.get('files_generated', []))
                    })
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x['start_time'], reverse=True)
        
        print(f"\n📁 Found {len(sessions)} sessions:")
        for session in sessions:
            print(f"  • {session['name']}: {session['tools_used']} tools, {session['files_generated']} files")
        
        return sessions

# Global session manager instance
_current_session = None

def get_session_manager(base_dir: str = "suite") -> SessionManager:
    """Get or create the current session manager"""
    global _current_session
    
    if _current_session is None:
        _current_session = SessionManager(base_dir)
    
    return _current_session

def finalize_current_session():
    """Finalize the current session"""
    global _current_session
    
    if _current_session is not None:
        session_dir = _current_session.finalize_session()
        _current_session = None
        return session_dir
    
    return None

# Convenience functions for easy use in other scripts
def save_session_image(fig, filename: str, category: str = 'images', **kwargs):
    """Convenience function to save image in current session"""
    return get_session_manager().save_image(fig, filename, category, **kwargs)

def save_session_data(data, filename: str, category: str = 'data', format: str = 'json'):
    """Convenience function to save data in current session"""
    return get_session_manager().save_data(data, filename, category, format)

def save_session_log(content: str, filename: str = None, append: bool = False):
    """Convenience function to save log in current session"""
    return get_session_manager().save_log(content, filename, append)

def save_session_report(content: str, filename: str, format: str = 'txt'):
    """Convenience function to save report in current session"""
    return get_session_manager().save_report(content, filename, format)

def add_session_note(note: str):
    """Convenience function to add note to current session"""
    return get_session_manager().add_note(note)

def copy_session_file(source_path: str, category: str, new_filename: str = None):
    """Convenience function to copy file to current session"""
    return get_session_manager().copy_file(source_path, category, new_filename)

if __name__ == "__main__":
    # Test the session manager
    sm = SessionManager()
    
    # Test various save operations
    sm.add_tool_usage("test_tool", "Testing session manager")
    sm.add_model_test("test_model", {"speed": 0.2, "smoothness": 0.85})
    sm.add_note("Testing session manager functionality")
    
    # Save some test data
    sm.save_data({"test": "data"}, "test_data.json")
    sm.save_log("Test log entry")
    
    # Finalize session
    final_dir = sm.finalize_session()
    print(f"\n🎯 Test session created at: {final_dir}")