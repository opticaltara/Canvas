"""create_uploaded_files_table

Revision ID: 4c1ed825800b
Revises: 42e41aaddcb5
Create Date: 2025-05-13 14:24:33.059111

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
# Import UUID type for id columns if your database supports it directly (e.g., PostgreSQL)
# For SQLite, SQLAlchemy handles UUIDs as strings, so direct import might not be needed
# from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '4c1ed825800b'
down_revision: Union[str, None] = '42e41aaddcb5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'uploaded_files',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('(LOWER(HEX(RANDOMBLOB(16))))') if op.get_bind().dialect.name == 'sqlite' else sa.text('gen_random_uuid()')),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('notebook_id', sa.UUID(), nullable=True),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('filepath', sa.String(), nullable=False, unique=True),
        sa.Column('file_type', sa.String(), nullable=True),
        sa.Column('size', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', sa.JSON(), nullable=True)
    )
    op.create_index(op.f('ix_uploaded_files_session_id'), 'uploaded_files', ['session_id'], unique=False)
    op.create_index(op.f('ix_uploaded_files_notebook_id'), 'uploaded_files', ['notebook_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_uploaded_files_notebook_id'), table_name='uploaded_files')
    op.drop_index(op.f('ix_uploaded_files_session_id'), table_name='uploaded_files')
    op.drop_table('uploaded_files')
