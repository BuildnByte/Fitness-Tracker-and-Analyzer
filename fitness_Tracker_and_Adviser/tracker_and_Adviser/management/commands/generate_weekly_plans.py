
from django.core.management.base import BaseCommand
from django.utils import timezone
from tracker_and_Adviser.views import generate_all_weekly_plans  # Import your function

class Command(BaseCommand):
    help = 'Generate weekly diet and workout plans for all users'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--user-id',
            type=str,
            help='Generate plan for specific user only',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force generation even if plans already exist',
        )
    
    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS(f'Starting weekly plan generation at {timezone.now()}')
        )
        
        if options['user_id']:
            # Generate for specific user
            from tracker_and_Adviser.views import generate_weekly_plan_for_user
            success = generate_weekly_plan_for_user(options['user_id'])
            if success:
                self.stdout.write(
                    self.style.SUCCESS(f'Successfully generated plan for user {options["user_id"]}')
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f'Failed to generate plan for user {options["user_id"]}')
                )
        else:
            # Generate for all users
            generate_all_weekly_plans()
            self.stdout.write(
                self.style.SUCCESS('Weekly plan generation completed for all users')
            )