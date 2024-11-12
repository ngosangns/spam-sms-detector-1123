import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { DialogComponent } from './dialog.component';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    FormsModule,
    CommonModule,
    MatInputModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    MatDialogModule,
    MatSnackBarModule,
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title = 'spam-detector-web';

  sms: string = '';
  isSMSLoading: boolean = false;

  url: string = '';
  isURLLoading: boolean = false;

  constructor(private dialog: MatDialog, private snackBar: MatSnackBar) {}

  showDialog(type: string, percent: number) {
    this.dialog.open(DialogComponent, {
      data: { type, percent },
    });
  }

  async onSMSSubmit() {
    this.isSMSLoading = true;
    await fetch('/sms/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ sms: this.sms }),
    })
      .then((response) => response.json())
      .then((data) => {
        this.showDialog(data.type, parseFloat(data.spam_percent));
      })
      .catch((error) => {
        console.error('Error:', error);
        this.snackBar.open('Failed to classify SMS', 'Close', {
          duration: 3000,
        });
      })
      .finally(() => {
        this.isSMSLoading = false;
      });
  }

  async onURLSubmit() {
    this.isURLLoading = true;
    await fetch('/url/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: this.url }),
    })
      .then((response) => response.json())
      .then((data) => {
        this.showDialog(data.type, parseFloat(data.spam_percent));
      })
      .catch((error) => {
        console.error('Error:', error);
        this.snackBar.open('Failed to classify URL', 'Close', {
          duration: 3000,
        });
      })
      .finally(() => {
        this.isURLLoading = false;
      });
  }
}
